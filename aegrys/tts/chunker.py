"""Split a streaming token feed into speakable chunks.

Why clauses and not sentences: at the measured ~12.7 tok/s, waiting for a full
sentence costs ~1.3 s before any audio plays. Piper synthesizes at RTF 0.15, so it
can start on a 4-word phrase. Emitting at the first clause boundary pulls several
hundred ms out of TTFA (DESIGN §7, revised by S2b/S5c).

The first chunk is deliberately eager -- it is the only one on the TTFA path.
Later chunks are allowed to be longer, because they are synthesized while earlier
audio is still playing.
"""

from __future__ import annotations

import re

SENTENCE_END = re.compile(r"[.!?]")
CLAUSE_END = re.compile(r"[,;:]")


class ClauseChunker:
    def __init__(self, min_words: int = 4, later_min_words: int = 8):
        self.min_words = min_words
        self.later_min_words = later_min_words
        self._buf = ""
        self._emitted = 0

    def _boundary(self, text: str) -> int:
        """Index just past the best split point, or -1."""
        need = self.min_words if self._emitted == 0 else self.later_min_words
        # Sentence end always wins.
        best = -1
        for m in SENTENCE_END.finditer(text):
            if len(text[:m.end()].split()) >= need:
                best = m.end()
                break
        if best != -1:
            return best
        # Only the first chunk is eager enough to break on a clause -- it's the
        # one on the TTFA path.
        if self._emitted == 0:
            for m in CLAUSE_END.finditer(text):
                if len(text[:m.end()].split()) >= need:
                    return m.end()
        return -1

    def push(self, token: str) -> list[str]:
        """Feed a token; return zero or more chunks ready to synthesize."""
        self._buf += token
        out = []
        while True:
            i = self._boundary(self._buf)
            if i == -1:
                break
            chunk = self._buf[:i].strip()
            self._buf = self._buf[i:].lstrip()
            if chunk:
                out.append(chunk)
                self._emitted += 1
        return out

    def flush(self) -> list[str]:
        """Emit whatever is left at end of stream."""
        rest = self._buf.strip()
        self._buf = ""
        if rest:
            self._emitted += 1
            return [rest]
        return []

    def reset(self) -> None:
        self._buf = ""
        self._emitted = 0
