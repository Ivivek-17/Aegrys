"""Text-to-speech via Piper.

Chosen over Kokoro-82M on measurement, not preference (S1):
  - Kokoro fp32 best case RTF 1.055 -- it can only just keep up with playback while
    using the same 6 threads the LLM needs. Under load it underruns and stutters.
  - Kokoro int8 was 3-4x SLOWER than fp32 (quantized ops falling off the optimized
    kernel path). Never assume int8 is faster on CPU.
  - Piper: RTF 0.15, and 0.195 even under full concurrent load (S5c) -- 5x margin
    against the RTF 1.0 underrun threshold.

Windows note: espeak-ng returns IPA and the default cp1252 console raises
UnicodeEncodeError on characters like U+025B. Set PYTHONIOENCODING=utf-8.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from ..core.config import TTSConfig


class Synthesizer:
    def __init__(self, cfg: TTSConfig):
        from piper import PiperVoice
        self.cfg = cfg
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.voice = PiperVoice.load(str(cfg.model_path))
        # PiperVoice.load gives no threading control, so swap in our own session.
        self.voice.session = ort.InferenceSession(
            str(cfg.model_path), so, providers=["CPUExecutionProvider"])
        self.sample_rate = self.voice.config.sample_rate
        self.synthesize("Ready.")   # warm

    def synthesize(self, text: str) -> np.ndarray:
        """Return float32 mono at self.sample_rate."""
        text = text.strip()
        if not text:
            return np.zeros(0, dtype=np.float32)
        parts = []
        for chunk in self.voice.synthesize(text):
            parts.append(np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16))
        if not parts:
            return np.zeros(0, dtype=np.float32)
        return (np.concatenate(parts).astype(np.float32) / 32768.0)
