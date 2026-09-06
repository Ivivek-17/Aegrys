"""Audio capture and playback.

Two hard requirements from Phase 0 / the design:

1. Capture at the device-native rate (48 kHz) and downsample in software. WASAPI
   will not resample for us (S3).
2. NEVER hand the output device more than ~30 ms of audio (DESIGN §4.2). If you
   write 2 seconds to the device, cancelling generation does not stop the sound --
   the OS already has it. Playback is fed from a deque that a barge-in can flush,
   so the worst-case audible tail after a cancel is one callback period.
"""

from __future__ import annotations

import queue
import threading
from collections import deque

import numpy as np
import sounddevice as sd

from ..core.config import AudioConfig
from .resample import Decimator, Interpolator


class MicStream:
    """Captures at device rate, yields fixed-size 16 kHz frames for the VAD."""

    def __init__(self, cfg: AudioConfig, frame_size: int):
        self.cfg = cfg
        self.frame_size = frame_size
        self._dec = Decimator(cfg.device_sr // cfg.model_sr)
        self._q: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=64)
        self._buf = np.zeros(0, dtype=np.float32)
        self._stream: sd.InputStream | None = None
        # Level of the most recent frame; the echo guard reads this.
        self.last_rms = 0.0

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # overflows are visible in the trace as jitter; don't spam stdout
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass  # drop rather than block the audio thread

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.cfg.device_sr,
            blocksize=int(self.cfg.device_sr * self.cfg.in_block_ms / 1000),
            channels=1, dtype="float32",
            device=self.cfg.in_device, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._q.put(None)

    def frames(self):
        """Yield exactly `frame_size` samples at 16 kHz (Silero needs exactly 512)."""
        while True:
            block = self._q.get()
            if block is None:
                return
            self._buf = np.concatenate([self._buf, self._dec(block)])
            while len(self._buf) >= self.frame_size:
                frame = self._buf[:self.frame_size]
                self._buf = self._buf[self.frame_size:]
                self.last_rms = float(np.sqrt(np.mean(frame ** 2)))
                yield frame


class SpeakerStream:
    """Chunked playback with instant flush.

    The deque holds device-rate chunks. `flush()` drops everything not yet handed
    to the driver, which is what makes barge-in feel immediate.
    """

    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self._interp = Interpolator(cfg.device_sr // cfg.model_sr)
        self._chunks: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream: sd.OutputStream | None = None
        self._blocksize = int(cfg.device_sr * cfg.out_chunk_ms / 1000)
        self._pending = np.zeros(0, dtype=np.float32)
        self._playing = threading.Event()
        # RMS of what we are currently emitting -- the echo guard compares the mic
        # against this to decide whether it is hearing itself (DESIGN §4.3).
        self.output_rms = 0.0

    def _callback(self, outdata, frames, time_info, status):
        with self._lock:
            while len(self._pending) < frames and self._chunks:
                self._pending = np.concatenate([self._pending, self._chunks.popleft()])
            if len(self._pending) >= frames:
                out = self._pending[:frames]
                self._pending = self._pending[frames:]
            else:
                out = np.zeros(frames, dtype=np.float32)
                if len(self._pending):
                    out[:len(self._pending)] = self._pending
                    self._pending = np.zeros(0, dtype=np.float32)
                self._playing.clear()
        self.output_rms = float(np.sqrt(np.mean(out ** 2)))
        outdata[:, 0] = out

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.cfg.device_sr, blocksize=self._blocksize,
            channels=1, dtype="float32",
            device=self.cfg.out_device, callback=self._callback)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def write(self, samples16k: np.ndarray) -> None:
        """Queue model-rate audio for playback, split into small chunks."""
        up = self._interp(samples16k)
        with self._lock:
            for i in range(0, len(up), self._blocksize):
                self._chunks.append(up[i:i + self._blocksize])
            self._playing.set()

    def flush(self) -> int:
        """Drop all queued audio. Returns milliseconds discarded."""
        with self._lock:
            n = sum(len(c) for c in self._chunks) + len(self._pending)
            self._chunks.clear()
            self._pending = np.zeros(0, dtype=np.float32)
            self._playing.clear()
            self._interp.reset()
        return int(n / self.cfg.device_sr * 1000)

    @property
    def is_playing(self) -> bool:
        return self._playing.is_set()

    def queued_ms(self) -> float:
        with self._lock:
            n = sum(len(c) for c in self._chunks) + len(self._pending)
        return n / self.cfg.device_sr * 1000

    def wait_drained(self, timeout: float = 30.0) -> None:
        self._playing.wait(0)
        import time
        t0 = time.perf_counter()
        while self.queued_ms() > 0 and time.perf_counter() - t0 < timeout:
            time.sleep(0.01)


def list_devices() -> str:
    lines = []
    hostapis = [a["name"] for a in sd.query_hostapis()]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] or d["max_output_channels"]:
            lines.append(f"[{i:>2}] in={d['max_input_channels']} "
                         f"out={d['max_output_channels']} "
                         f"{hostapis[d['hostapi']][:14]:<14} {d['name'][:46]}")
    return "\n".join(lines)
