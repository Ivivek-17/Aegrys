"""Streaming sample-rate conversion between device rate and model rate.

S3 finding: WASAPI (shared mode, via PortAudio) will NOT resample. Opening the mic
at 16 kHz raises "Invalid sample rate [PaErrorCode -9997]" -- devices here are
natively 48 kHz. So we capture at 48 k and convert in software, both ways.

48000/16000 is exactly 3, so this is a clean 3:1 polyphase decimator / interpolator.
Both keep filter state across blocks, which matters: a stateless per-block filter
produces a click at every block boundary.

numpy-only on purpose -- scipy would be ~40 MB for one function.
"""

from __future__ import annotations

import numpy as np

RATIO = 3


def _lowpass(ntaps: int, cutoff: float) -> np.ndarray:
    """Windowed-sinc FIR. `cutoff` is normalized to the sample rate (0..0.5)."""
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = 2 * cutoff * np.sinc(2 * cutoff * n)
    h *= np.hamming(ntaps)
    return (h / h.sum()).astype(np.float32)


class Decimator:
    """48 kHz -> 16 kHz. Anti-alias filter then take every 3rd sample."""

    def __init__(self, ratio: int = RATIO, ntaps: int = 63):
        self.ratio = ratio
        # Cutoff just below the output Nyquist (16k/2 = 8k -> 8k/48k = 0.1667).
        # Pull in slightly to leave transition-band room.
        self.taps = _lowpass(ntaps, 0.45 / ratio)
        self._tail = np.zeros(len(self.taps) - 1, dtype=np.float32)
        self._phase = 0

    def __call__(self, block: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._tail, block.astype(np.float32, copy=False)])
        # 'valid' consumes the overlap; keep the tail for the next call.
        y = np.convolve(x, self.taps, mode="valid")
        self._tail = x[-(len(self.taps) - 1):] if len(self.taps) > 1 else x[:0]
        # Keep decimation phase continuous across blocks.
        out = y[self._phase::self.ratio]
        consumed = len(y)
        self._phase = (self._phase - consumed) % self.ratio
        return out.astype(np.float32, copy=False)

    def reset(self) -> None:
        self._tail[:] = 0
        self._phase = 0


class Interpolator:
    """16 kHz -> 48 kHz. Zero-stuff then lowpass to suppress the images."""

    def __init__(self, ratio: int = RATIO, ntaps: int = 63):
        self.ratio = ratio
        self.taps = _lowpass(ntaps, 0.45 / ratio) * ratio  # gain compensation
        self._tail = np.zeros(len(self.taps) - 1, dtype=np.float32)

    def __call__(self, block: np.ndarray) -> np.ndarray:
        up = np.zeros(len(block) * self.ratio, dtype=np.float32)
        up[::self.ratio] = block.astype(np.float32, copy=False)
        x = np.concatenate([self._tail, up])
        y = np.convolve(x, self.taps, mode="valid")
        self._tail = x[-(len(self.taps) - 1):] if len(self.taps) > 1 else x[:0]
        return y.astype(np.float32, copy=False)

    def reset(self) -> None:
        self._tail[:] = 0


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """One-shot convenience for offline/test paths."""
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    if src_sr % dst_sr == 0:
        return Decimator(src_sr // dst_sr)(x)
    if dst_sr % src_sr == 0:
        return Interpolator(dst_sr // src_sr)(x)
    n = int(round(len(x) * dst_sr / src_sr))
    return np.interp(np.linspace(0, len(x) - 1, n),
                     np.arange(len(x)), x).astype(np.float32)
