"""Unit tests for the pieces barge-in depends on.

These run without audio hardware or an LLM. The invariants here are the ones that,
if broken, make barge-in silently unreliable -- which is the failure mode hardest
to notice in a demo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aegrys.audio.io import SpeakerStream
from aegrys.core.barge import EchoGuard
from aegrys.core.config import AudioConfig, Config, VADConfig
from aegrys.core.epoch import Cancelled, EpochController
from aegrys.tts.chunker import ClauseChunker
from aegrys.vad.endpointer import Endpointer, State


# --------------------------------------------------------------------- epochs

def test_epoch_begin_increments_and_invalidates_previous():
    e = EpochController()
    a = e.begin()
    b = e.begin()
    assert b > a
    assert e.is_current(b) and not e.is_current(a)


def test_cancel_invalidates_current_epoch():
    e = EpochController()
    a = e.begin()
    assert e.is_current(a)
    e.cancel()
    assert not e.is_current(a), "cancelled epoch must not remain current"


def test_check_raises_for_stale_epoch():
    e = EpochController()
    a = e.begin()
    e.cancel()
    with pytest.raises(Cancelled):
        e.check(a)


def test_guard_stops_generator_at_cancellation():
    """A token stream must stop feeding TTS the moment the epoch goes stale."""
    e = EpochController()
    epoch = e.begin()
    seen = []

    def tokens():
        for i in range(100):
            yield i

    for i in e.guard(epoch)(tokens()):
        seen.append(i)
        if i == 4:
            e.cancel()
    assert seen == [0, 1, 2, 3, 4], "stale tokens leaked past cancellation"


# -------------------------------------------------------------------- speaker

def _cfg_audio():
    return AudioConfig(device_sr=48000, model_sr=16000, out_chunk_ms=30)


def test_speaker_flush_drops_all_queued_audio():
    s = SpeakerStream(_cfg_audio())
    s.write(np.ones(16000, dtype=np.float32) * 0.1)   # 1 second
    assert s.queued_ms() > 900
    dropped = s.flush()
    assert dropped > 900
    assert s.queued_ms() == 0, "flush must leave nothing queued"


def test_speaker_chunks_are_bounded():
    """Never hand the device more than one chunk, or barge-in leaves a tail."""
    cfg = _cfg_audio()
    s = SpeakerStream(cfg)
    s.write(np.ones(16000, dtype=np.float32) * 0.1)
    expected = int(cfg.device_sr * cfg.out_chunk_ms / 1000)
    assert all(len(c) <= expected for c in s._chunks)


# -------------------------------------------------------------------- chunker

def test_chunker_emits_first_chunk_on_clause_for_low_latency():
    """A clause past the word floor should be emitted before the sentence ends."""
    c = ClauseChunker(min_words=4)
    out = []
    for tok in "I have set that timer, and your meeting is at four. ".split(" "):
        out += c.push(tok + " ")
    assert out, "expected an early chunk at the comma"
    assert out[0].endswith(","), f"first chunk should break on the clause: {out[0]}"
    out += c.flush()
    assert len(out) > 1, "remainder should follow as its own chunk"


def test_chunker_waits_when_first_clause_is_below_word_floor():
    """A 2-word clause is too short to speak on its own; wait for the sentence."""
    c = ClauseChunker(min_words=4)
    out = []
    for tok in "Sure thing, I will set that up for you. ".split(" "):
        out += c.push(tok + " ")
    assert len(out) == 1 and out[0].endswith(".")


def test_chunker_does_not_emit_tiny_fragments():
    c = ClauseChunker(min_words=4)
    out = []
    for tok in ["Hi", ",", " there", ","]:
        out += c.push(tok)
    assert out == [], "must not emit a chunk below the word floor"


def test_chunker_flush_returns_remainder():
    c = ClauseChunker(min_words=4)
    c.push("no terminal punctuation here")
    assert c.flush() == ["no terminal punctuation here"]
    assert c.flush() == [], "flush must be idempotent"


def test_chunker_full_text_is_preserved():
    text = "First part, second part. Third part! Fourth?"
    c = ClauseChunker(min_words=2)
    out = []
    for ch in text:
        out += c.push(ch)
    out += c.flush()
    joined = " ".join(out)
    assert joined.replace(" ", "") == text.replace(" ", ""), \
        f"chunker lost or reordered text: {joined!r}"


# ------------------------------------------------------------------ endpointer

def _run(ep, speech_frames, silence_frames):
    f = np.zeros(512, dtype=np.float32)
    st = State.IDLE
    for _ in range(speech_frames):
        st = ep.push(f, True)
    for _ in range(silence_frames):
        st = ep.push(f, False)
        if st == State.ENDPOINTED:
            break
    return st


def test_endpointer_waits_longer_after_a_trailing_word():
    cfg = VADConfig()
    ep = Endpointer(cfg, frame_ms=32)
    ep.set_hint("remind me to call mom and")     # ends in a conjunction
    st = _run(ep, speech_frames=20, silence_frames=int(500 / 32))
    assert st != State.ENDPOINTED, "should still be waiting mid-thought"


def test_endpointer_fires_quickly_on_complete_sentence():
    cfg = VADConfig()
    ep = Endpointer(cfg, frame_ms=32)
    ep.set_hint("what time is it?")
    st = _run(ep, speech_frames=20, silence_frames=int(450 / 32) + 2)
    assert st == State.ENDPOINTED


def test_endpointer_ignores_short_blips():
    cfg = VADConfig()
    ep = Endpointer(cfg, frame_ms=32)
    st = _run(ep, speech_frames=2, silence_frames=int(1200 / 32))
    assert st != State.ENDPOINTED, "a 64ms blip must not open a turn"


# ------------------------------------------------------------------ echo guard

def test_echo_guard_suppresses_our_own_playback():
    g = EchoGuard(Config(), frame_ms=32)
    # Mic hears roughly what we're playing -> that's us, not the user.
    for _ in range(50):
        assert not g.allows(True, mic_rms=0.10, out_rms=0.10, playing=True)


def test_echo_guard_requires_sustained_speech():
    g = EchoGuard(Config(), frame_ms=32)
    loud = dict(mic_rms=1.0, out_rms=0.05, playing=True)
    # A couple of loud frames is not enough; 300ms sustained is.
    assert not g.allows(True, **loud)
    assert not g.allows(True, **loud)
    fired = any(g.allows(True, **loud) for _ in range(20))
    assert fired, "sustained loud speech must eventually barge in"


def test_echo_guard_resets_on_gap():
    g = EchoGuard(Config(), frame_ms=32)
    loud = dict(mic_rms=1.0, out_rms=0.05, playing=True)
    for _ in range(5):
        g.allows(True, **loud)
    g.allows(False, **loud)          # a gap
    assert not g.allows(True, **loud), "sustain counter must reset on a gap"


def test_echo_guard_is_permissive_when_not_playing():
    g = EchoGuard(Config(), frame_ms=32)
    assert g.allows(True, mic_rms=0.01, out_rms=0.0, playing=False)


# ------------------------------------------------------------------------ vad

def test_vad_detects_real_speech():
    """Regression: Silero v5 needs 64 samples of CONTEXT prepended to each frame.

    Without it the model does not error -- it silently returns ~0.001 for clear
    human speech, which made voice input detect nothing at all. This test is the
    guard: it must FAIL if the context buffer is ever removed.
    """
    import soundfile as sf
    from aegrys.core.config import VADConfig
    from aegrys.vad.silero import SileroVAD

    wav = Path(__file__).resolve().parents[1] / "bench" / "jfk.wav"
    if not wav.exists():
        pytest.skip("bench/jfk.wav fixture not present")
    audio, sr = sf.read(wav, dtype="float32")
    assert sr == 16000

    vad = SileroVAD(VADConfig())
    probs = [vad(audio[i:i + 512]) for i in range(0, len(audio) - 512, 512)]
    speech = sum(p > 0.5 for p in probs)
    assert max(probs) > 0.9, (
        f"VAD failed to detect clear speech (max prob {max(probs):.3f}). "
        "The 64-sample context buffer is probably missing.")
    assert speech > len(probs) * 0.4, (
        f"only {speech}/{len(probs)} frames detected as speech")


def test_vad_rejects_silence():
    from aegrys.core.config import VADConfig
    from aegrys.vad.silero import SileroVAD

    vad = SileroVAD(VADConfig())
    probs = [vad(np.zeros(512, dtype=np.float32)) for _ in range(100)]
    assert max(probs) < 0.5, f"VAD fired on silence (max {max(probs):.3f})"
