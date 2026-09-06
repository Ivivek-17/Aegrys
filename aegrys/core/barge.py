"""Phase 2 — barge-in.

Three things have to be true for an interruption to feel instant:

1. Generation stops. Handled by bumping the epoch: the LLM stream, the TTS loop and
   any tool call all check `is_current` and abandon quietly (DESIGN §4.1).
2. The sound stops. Handled by flushing the speaker deque. Because we never hand the
   device more than one chunk (~30 ms), that is the worst-case audible tail (§4.2).
3. We don't interrupt ourselves. That's the echo guard below, and it is the part
   most likely to be wrong on this hardware -- R1 is still unresolved (S3 could not
   distinguish "hardware AEC" from "the speakers were muted").

The listener runs on its own thread and never blocks on the LLM or TTS, so speech
is detected even while the rest of the pipeline is busy. S5c measured VAD jitter at
5.3 ms worst case against a 32 ms frame budget, so this holds under full load.
"""

from __future__ import annotations

import queue
import threading

import numpy as np

from ..vad.endpointer import Endpointer, State
from .config import VAD_FRAME, Config
from .trace import Tracer
from .turn import Assistant


class EchoGuard:
    """Decides whether mic energy during playback is the user or ourselves.

    Baseline mitigation from DESIGN §4.3: require BOTH sustained speech and mic
    energy meaningfully above what our own playback is producing. Deliberately
    conservative -- a false barge-in (cutting ourselves off mid-word for no reason)
    is far more damaging to a demo than a missed one.
    """

    def __init__(self, cfg: Config, frame_ms: float):
        self.cfg = cfg
        self.frame_ms = frame_ms
        self._sustained_ms = 0.0
        # Mic must exceed playback level by this much to count as real speech.
        self.margin = 2.0

    def reset(self) -> None:
        self._sustained_ms = 0.0

    def allows(self, is_speech: bool, mic_rms: float, out_rms: float,
               playing: bool) -> bool:
        if not self.cfg.echo_guard or not playing:
            # Not our audio: a single speech frame is enough.
            self._sustained_ms = self._sustained_ms + self.frame_ms if is_speech else 0.0
            return is_speech
        if not is_speech or mic_rms < out_rms * self.margin:
            self._sustained_ms = 0.0
            return False
        self._sustained_ms += self.frame_ms
        return self._sustained_ms >= self.cfg.echo_guard_sustain_ms


class BargeInAssistant(Assistant):
    def __init__(self, cfg: Config, tracer: Tracer | None = None):
        super().__init__(cfg, tracer)
        self.frame_ms = VAD_FRAME / cfg.audio.model_sr * 1000
        self.guard = EchoGuard(cfg, self.frame_ms)
        self._utterances: queue.Queue[np.ndarray] = queue.Queue()
        self._listener: threading.Thread | None = None
        self._barge_ins = 0

    # ------------------------------------------------------------- listener

    def start(self) -> None:
        super().start()
        self._listener = threading.Thread(target=self._listen_loop, daemon=True)
        self._listener.start()

    def _listen_loop(self) -> None:
        ep = Endpointer(self.cfg.vad, frame_ms=self.frame_ms)
        self.vad.reset()
        for frame in self.mic.frames():
            if not self._running:
                return
            speech = self.vad.is_speech(frame)
            playing = self.speaker.is_playing

            if playing:
                # While we're talking, only a guarded, sustained interruption counts.
                if self.guard.allows(speech, self.mic.last_rms,
                                     self.speaker.output_rms, True):
                    self._trigger_barge_in()
                    ep.reset()
                    self.vad.reset()
                    ep.push(frame, True)
                continue

            self.guard.reset()
            if ep.push(frame, speech) == State.ENDPOINTED:
                self.tracer.mark("speech_end")
                self._utterances.put(ep.utterance())
                ep.reset()
                self.vad.reset()

    def _trigger_barge_in(self) -> None:
        cancelled = self.epochs.cancel()
        dropped = self.speaker.flush()
        self._barge_ins += 1
        t = self.tracer.current
        if t is not None and t.turn_id == cancelled:
            t.cancelled = True
            t.meta["barge_in_dropped_ms"] = dropped
        print(f"  \033[31m⨯ barge-in\033[0m (dropped {dropped}ms of audio)",
              flush=True)

    # ----------------------------------------------------------------- loop

    def run_forever(self) -> None:
        self.start()
        print("\n\033[1mAegrys ready.\033[0m Speak any time — including while I'm "
              "talking. Ctrl-C to quit.\n", flush=True)
        try:
            while self._running:
                try:
                    audio = self._utterances.get(timeout=0.2)
                except queue.Empty:
                    continue
                if audio.size == 0:
                    continue
                # Anything still queued is stale the moment a new turn starts.
                while not self._utterances.empty():
                    try:
                        self._utterances.get_nowait()
                    except queue.Empty:
                        break
                self._one_turn(audio)
        except KeyboardInterrupt:
            print(f"\n{self.tracer.table()}\nbarge-ins: {self._barge_ins}",
                  flush=True)
        finally:
            self.stop()

    def _one_turn(self, audio: np.ndarray) -> None:
        epoch = self.epochs.begin()
        self.tracer.start_turn(epoch)
        text = self.transcribe(audio)
        if not text or len(text) < 2:
            return
        print(f"  \033[33myou:\033[0m {text}", flush=True)
        reply = self.respond(text, epoch)
        if reply:
            print(f"  \033[32maegrys:\033[0m {reply}", flush=True)
        self.tracer.current.meta["text"] = text
        self.tracer.end_turn()
