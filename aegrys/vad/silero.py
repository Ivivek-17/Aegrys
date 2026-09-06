"""Silero VAD (ONNX) — voice/no-voice on 32 ms frames.

ONNX + onnxruntime directly rather than the `silero-vad` pip package, which pulls
torch. We already run onnxruntime for TTS; a second framework would mean a second
thread pool competing for the same 12 threads (DESIGN §3).

Measured: 0.15 ms per frame idle, 2.9 ms p99 under full load (S5c) against a 32 ms
budget. It gets one dedicated thread and must never be starved.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from ..core.config import VADConfig


class SileroVAD:
    def __init__(self, cfg: VADConfig, sample_rate: int = 16000):
        so = ort.SessionOptions()
        so.intra_op_num_threads = cfg.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(cfg.model_path), so,
                                         providers=["CPUExecutionProvider"])
        self.cfg = cfg
        self._sr = np.array(sample_rate, dtype=np.int64)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        # Silero v5 needs 64 samples of CONTEXT from the previous chunk prepended
        # to each 512-sample frame (32 at 8 kHz), so the model actually consumes
        # 576 samples per step.
        #
        # This is not optional and it fails SILENTLY: feeding a bare 512 samples
        # returns ~0.001 for clear human speech instead of raising. Measured on
        # bench/jfk.wav -- 512 alone: max prob 0.004, 0/343 frames detected;
        # 64+512: max prob 1.000, 233/343 detected.
        self._ctx_size = 64 if sample_rate == 16000 else 32
        self._ctx = np.zeros(self._ctx_size, dtype=np.float32)

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(self._ctx_size, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        """Speech probability for one 512-sample frame at 16 kHz."""
        frame = frame.astype(np.float32, copy=False)
        x = np.concatenate([self._ctx, frame]).reshape(1, -1)
        self._ctx = frame[-self._ctx_size:].copy()
        out = self.sess.run(None, {"input": x, "state": self._state, "sr": self._sr})
        self._state = out[1]
        return float(out[0][0][0])

    def is_speech(self, frame: np.ndarray) -> bool:
        return self(frame) >= self.cfg.speech_threshold
