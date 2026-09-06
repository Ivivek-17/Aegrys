"""Speech-to-text via faster-whisper.

S4 findings baked in here:
  - Whisper pads every input to a fixed 30 s mel window, so decode cost is
    ENCODER-dominated and roughly CONSTANT regardless of utterance length
    (3s -> 2641 ms, 10s -> 2795 ms for the same model). Do not expect short
    utterances to be cheap.
  - Because of that, `distil-*` models are the wrong family on CPU: they shrink the
    decoder. distil-small.en measured 3.5x SLOWER than plain base.en.
  - `chunk_length` is not a lever; shortening the window made it slower.
  - Each partial costs a FULL encoder pass, so continuous partials are not
    affordable. Default is endpoint-then-decode.
"""

from __future__ import annotations

import numpy as np

from ..core.config import STTConfig


class Transcriber:
    def __init__(self, cfg: STTConfig):
        from faster_whisper import WhisperModel
        self.cfg = cfg
        self.model = WhisperModel(
            cfg.model, device="cpu", compute_type=cfg.compute_type,
            cpu_threads=cfg.threads, num_workers=1)
        self._warm()

    def _warm(self) -> None:
        """First decode pays lazy init; do it at startup, not on turn 1."""
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, _ = self.model.transcribe(
            audio.astype(np.float32, copy=False),
            beam_size=self.cfg.beam_size,
            language="en",
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        # transcribe() returns a generator; it must be drained to finish decoding.
        return " ".join(s.text for s in segments).strip()
