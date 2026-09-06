"""Turn epochs — the cancellation primitive (DESIGN §4.1).

A boolean stop-flag cannot express "drop work that is already in flight": an HTTP
response mid-stream, a TTS chunk already handed to the audio device, an MCP call
already dispatched. There is always a race between checking the flag and emitting
the result.

Instead every turn gets a monotonically increasing id. Work carries the id it was
started for, and every stage boundary drops anything stale. No stage needs to know
*why* it was cancelled, and a late arrival is simply ignored.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class Cancelled(Exception):
    """Raised when work discovers its epoch is no longer current."""

    def __init__(self, epoch: int, current: int):
        super().__init__(f"epoch {epoch} cancelled (current is {current})")
        self.epoch = epoch
        self.current = current


@dataclass(frozen=True)
class Tagged(Generic[T]):
    """A payload stamped with the epoch it belongs to."""

    epoch: int
    payload: T


class EpochController:
    """Owns the current epoch. Thread-safe; every stage shares one instance."""

    def __init__(self) -> None:
        self._epoch = 0
        self._lock = threading.Lock()
        # Set while an epoch is live; cleared and replaced on cancel so blocking
        # waiters wake immediately rather than polling.
        self._alive = threading.Event()
        self._alive.set()

    @property
    def current(self) -> int:
        with self._lock:
            return self._epoch

    def begin(self) -> int:
        """Start a new turn. Implicitly cancels anything still running."""
        with self._lock:
            self._epoch += 1
            self._alive = threading.Event()
            self._alive.set()
            return self._epoch

    def cancel(self) -> int:
        """Invalidate the current epoch. Returns the epoch that was cancelled."""
        with self._lock:
            cancelled = self._epoch
            self._epoch += 1
            self._alive.clear()
            self._alive = threading.Event()
            self._alive.set()
            return cancelled

    def is_current(self, epoch: int) -> bool:
        with self._lock:
            return epoch == self._epoch

    def check(self, epoch: int) -> None:
        """Raise if `epoch` is stale. Call at loop tops and before side effects."""
        with self._lock:
            if epoch != self._epoch:
                raise Cancelled(epoch, self._epoch)

    def guard(self, epoch: int):
        """Filter a generator, stopping cleanly as soon as the epoch goes stale.

        Used to wrap token streams so a barge-in ends iteration instead of
        letting stale tokens reach the TTS stage.
        """

        def wrap(it):
            for item in it:
                if not self.is_current(epoch):
                    return
                yield item

        return wrap
