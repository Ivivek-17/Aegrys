"""Endpointing — deciding the user finished their thought (DESIGN §5.1).

VAD answers "voice or not". It does NOT answer "are they done talking". That's this
module, and it owns a real tradeoff: fire early and you cut people off; fire late
and every turn feels sluggish.

Adaptive silence threshold driven by a zero-cost heuristic on the text so far.
No extra model, no extra compute -- which matters because S4 showed every STT pass
costs a full encoder run, so we cannot afford to think hard here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..core.config import VADConfig

# Ending on one of these means the speaker is mid-thought; wait longer.
TRAILING = {
    "and", "but", "or", "so", "because", "if", "when", "while", "that", "which",
    "the", "a", "an", "to", "for", "with", "at", "in", "on", "of", "my", "your",
    "um", "uh", "erm", "like", "well", "then", "is", "was", "are", "i",
}
COMPLETE_RE = re.compile(r"[.!?]\s*$")


class State(Enum):
    IDLE = "idle"
    SPEAKING = "speaking"
    TRAILING_SILENCE = "trailing"
    ENDPOINTED = "endpointed"


@dataclass
class Endpointer:
    cfg: VADConfig
    frame_ms: float = 32.0
    state: State = State.IDLE
    speech_ms: float = 0.0
    silence_ms: float = 0.0
    total_ms: float = 0.0
    _frames: list[np.ndarray] = field(default_factory=list)
    _hint: str = ""

    def reset(self) -> None:
        self.state = State.IDLE
        self.speech_ms = self.silence_ms = self.total_ms = 0.0
        self._frames.clear()
        self._hint = ""

    def set_hint(self, text: str) -> None:
        """Optional partial transcript, used only to pick the silence threshold."""
        self._hint = text or ""

    def _required_silence_ms(self) -> float:
        t = self._hint.strip().lower()
        if not t:
            return self.cfg.silence_default_ms
        if COMPLETE_RE.search(t):
            return self.cfg.silence_complete_ms
        last = re.sub(r"[^a-z']", "", t.split()[-1]) if t.split() else ""
        if last in TRAILING:
            return self.cfg.silence_trailing_ms
        return self.cfg.silence_default_ms

    def push(self, frame: np.ndarray, is_speech: bool) -> State:
        """Feed one frame. Returns the state; ENDPOINTED means the turn is ready."""
        self.total_ms += self.frame_ms

        if is_speech:
            self.silence_ms = 0.0
            self.speech_ms += self.frame_ms
            self._frames.append(frame)
            if self.state in (State.IDLE, State.TRAILING_SILENCE):
                self.state = State.SPEAKING
        else:
            if self.state == State.IDLE:
                # Keep a little pre-roll so we don't clip the first phoneme.
                self._frames.append(frame)
                if len(self._frames) > int(300 / self.frame_ms):
                    self._frames.pop(0)
                return self.state
            self.silence_ms += self.frame_ms
            self._frames.append(frame)
            if self.state == State.SPEAKING:
                self.state = State.TRAILING_SILENCE

        if (self.state == State.TRAILING_SILENCE
                and self.speech_ms >= self.cfg.min_speech_ms
                and self.silence_ms >= self._required_silence_ms()):
            self.state = State.ENDPOINTED
        elif self.total_ms >= self.cfg.max_utterance_ms and self.speech_ms > 0:
            self.state = State.ENDPOINTED

        return self.state

    def utterance(self) -> np.ndarray:
        if not self._frames:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._frames)

    @property
    def has_speech(self) -> bool:
        return self.speech_ms >= self.cfg.min_speech_ms
