"""Central configuration.

Every default here that carries a number was MEASURED in Phase 0, not guessed.
See bench/RESULTS.md. Where a value contradicts intuition there's a comment saying
why, so nobody "optimizes" it back to the intuitive-but-slower setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CACHE = Path(os.environ.get("AEGRYS_CACHE", "D:/Aegrys/.cache"))

# Audio contract: models want 16 kHz mono float32; WASAPI devices are 48 kHz and
# PortAudio will NOT resample for us (S3: "Invalid sample rate" at 16k).
MODEL_SR = 16000
VAD_FRAME = 512          # Silero v5 requires exactly 512 samples @ 16 kHz (32 ms)


@dataclass
class AudioConfig:
    model_sr: int = MODEL_SR
    device_sr: int = 48000          # S3: WASAPI native rate; we downsample in software
    in_device: int | None = None    # None = system default
    out_device: int | None = None
    # Output is fed in small chunks so a barge-in can flush it (DESIGN §4.2).
    # Never hand the device more than this, or cancelled speech keeps playing.
    out_chunk_ms: int = 30
    in_block_ms: int = 32           # matches the VAD frame period


@dataclass
class VADConfig:
    model_path: Path = CACHE / "vad" / "silero_vad.onnx"
    threads: int = 1                # S5c: 0.15 ms/frame; jitter 5.3 ms worst case
    speech_threshold: float = 0.5
    # Endpointing (DESIGN §5.1). VAD says voice/no-voice; these turn that into
    # "the user finished their thought".
    min_speech_ms: int = 200        # ignore blips
    silence_complete_ms: int = 400  # transcript looks finished
    silence_default_ms: int = 600
    silence_trailing_ms: int = 900  # ends in a conjunction/filler
    max_utterance_ms: int = 20000   # hard stop


@dataclass
class STTConfig:
    # S4: Whisper pads every input to a fixed 30s mel window, so cost is
    # ENCODER-dominated and constant. distil-* shrinks the DECODER, so it does not
    # help on CPU -- distil-small.en measured 3.5x SLOWER than base.en.
    model: str = "tiny.en"          # 294 ms @ 6 threads; base.en = 753 ms
    compute_type: str = "int8"
    threads: int = 4
    beam_size: int = 1
    # S4: each partial costs a FULL encoder pass, so continuous partials would
    # burn a 66% duty cycle stolen from the LLM. Endpoint-then-decode instead.
    partials_enabled: bool = False


@dataclass
class LLMConfig:
    host: str = os.environ.get("AEGRYS_LLM_HOST", "http://127.0.0.1:11435")
    model: str = "qwen2.5:3b-instruct-q4_K_M"
    # S2: 6 is optimal. 12 threads is 19% WORSE (Intel hybrid P/E cores).
    # S5b: ollama RELOADS the model if this changes between requests -- never sweep it.
    threads: int = 6
    keep_alive: str = "30m"
    temperature: float = 0.0
    num_predict: int = 160
    # S2b: at ~12.7 tok/s every emitted token costs ~79 ms, so response length is
    # a latency setting, not a style setting.
    max_sentence_words: int = 24


@dataclass
class TTSConfig:
    model_path: Path = CACHE / "piper" / "en_US-amy-low.onnx"
    # S1: Kokoro-82M measured RTF 1.055 (int8 was 3-4x SLOWER than fp32) -- it
    # cannot sustain real-time under load. Piper is RTF 0.15, 7x faster.
    threads: int = 4                # S5c: same LLM cost as 2, better RTF
    # DESIGN §7 / S5c: chunk on clauses, not sentences, so playback starts sooner.
    min_chunk_words: int = 4


@dataclass
class ToolsConfig:
    enabled: bool = True
    # "router": two small constrained calls (~270 prompt tokens total).
    # "native": bind all MCP schemas to one call (~900 prompt tokens).
    # S7/S8 measured native prefill at 15-20 s when the prefix cache misses on this
    # CPU (~50 tok/s), so router is the default HERE. On a GPU, native would win.
    mode: str = "router"
    call_timeout_s: float = 8.0
    db_path: Path = Path("D:/Aegrys/.cache/aegrys.db")


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    trace: bool = True
    # Barge-in (DESIGN §4.3). R1 is UNRESOLVED: the mic may hear the speakers.
    # Half-duplex gate is the baseline mitigation until a human test settles it.
    barge_in: bool = True
    echo_guard: bool = True
    echo_guard_sustain_ms: int = 300   # sustained speech required during playback


def load() -> Config:
    return Config()
