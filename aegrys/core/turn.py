"""Turn orchestrator — the control loop.

Deliberately hand-written rather than a LangChain AgentExecutor (DESIGN §6.6).
Agent frameworks cannot express mid-generation cancellation or streaming partial
output into TTS, and those two things ARE the product. This file is the place
where cancellation is explicit and readable.

Phase 1: half-duplex. listen -> transcribe -> respond -> speak, one at a time.
Barge-in arrives in Phase 2 (see BargeInAssistant).
"""

from __future__ import annotations

import numpy as np

from ..audio.io import MicStream, SpeakerStream
from ..llm.client import LLMClient
from ..llm.prompts import SYSTEM
from ..stt.whisper import Transcriber
from ..tts.chunker import ClauseChunker
from ..tts.piper import Synthesizer
from ..vad.endpointer import Endpointer, State
from ..vad.silero import SileroVAD
from .config import VAD_FRAME, Config
from .epoch import EpochController
from .trace import Tracer


class Assistant:
    def __init__(self, cfg: Config, tracer: Tracer | None = None):
        self.cfg = cfg
        self.tracer = tracer or Tracer(cfg.trace)
        self.epochs = EpochController()

        print("loading models...", flush=True)
        self.vad = SileroVAD(cfg.vad)
        self.stt = Transcriber(cfg.stt)
        self.tts = Synthesizer(cfg.tts)
        self.llm = LLMClient(cfg.llm, self.epochs)
        self.mic = MicStream(cfg.audio, VAD_FRAME)
        self.speaker = SpeakerStream(cfg.audio)
        self.history: list[dict] = [{"role": "system", "content": SYSTEM}]
        self._running = False

    # ------------------------------------------------------------------ io

    def start(self) -> None:
        self.speaker.start()
        self.mic.start()
        self._running = True

    def stop(self) -> None:
        self._running = False
        self.mic.stop()
        self.speaker.stop()

    def shutdown(self) -> None:
        """Release everything. Safe to call twice."""
        try:
            self.stop()
        except Exception:
            pass

    # --------------------------------------------------------------- listen

    def listen(self) -> np.ndarray:
        """Block until the endpointer says the user finished. Returns 16 kHz audio."""
        ep = Endpointer(self.cfg.vad, frame_ms=VAD_FRAME / self.cfg.audio.model_sr * 1000)
        self.vad.reset()
        announced = False
        for frame in self.mic.frames():
            if not self._running:
                break
            speech = self.vad.is_speech(frame)
            state = ep.push(frame, speech)
            if state == State.SPEAKING and not announced:
                print("  \033[36m● listening\033[0m", flush=True)
                announced = True
            if state == State.ENDPOINTED:
                self.tracer.mark("speech_end")
                return ep.utterance()
        return np.zeros(0, dtype=np.float32)

    # -------------------------------------------------------------- respond

    def transcribe(self, audio: np.ndarray) -> str:
        with self.tracer.span("stt", samples=len(audio)):
            text = self.stt.transcribe(audio)
        self.tracer.mark("stt_done")
        return text

    def respond(self, text: str, epoch: int) -> str:
        """Stream the reply, synthesizing and playing clause by clause."""
        self.history.append({"role": "user", "content": text})
        chunker = ClauseChunker(min_words=self.cfg.tts.min_chunk_words)
        spoken: list[str] = []

        def speak(chunk: str) -> None:
            with self.tracer.span("tts", chars=len(chunk)):
                audio = self.tts.synthesize(chunk)
            if not self.epochs.is_current(epoch):
                return
            self.tracer.mark("tts_first_chunk")
            self.speaker.write(audio)
            spoken.append(chunk)

        first = True
        for tok in self.llm.chat_stream(self.history, epoch=epoch):
            if first:
                self.tracer.mark("llm_first_token")
                first = False
            for chunk in chunker.push(tok):
                if not self.epochs.is_current(epoch):
                    return " ".join(spoken)
                self.tracer.mark("llm_first_sentence")
                speak(chunk)
        for chunk in chunker.flush():
            if not self.epochs.is_current(epoch):
                break
            self.tracer.mark("llm_first_sentence")
            speak(chunk)

        reply = " ".join(spoken)
        if reply:
            self.history.append({"role": "assistant", "content": reply})
        return reply

    # ----------------------------------------------------------------- loop

    def handle_text(self, text: str) -> str:
        """Text-in/text-out entry point. Used by the replay harness and tests."""
        epoch = self.epochs.begin()
        self.tracer.start_turn(epoch)
        self.tracer.mark("speech_end")
        self.tracer.mark("stt_done")
        print(f"  \033[33myou:\033[0m {text}", flush=True)
        reply = self.respond(text, epoch)
        print(f"  \033[32maegrys:\033[0m {reply}", flush=True)
        self.tracer.current.meta["text"] = text
        self.tracer.end_turn()
        return reply

    def run_forever(self) -> None:
        self.start()
        print("\n\033[1mAegrys ready.\033[0m Speak, or Ctrl-C to quit.\n", flush=True)
        try:
            while self._running:
                audio = self.listen()
                if audio.size == 0:
                    continue
                epoch = self.epochs.begin()
                self.tracer.start_turn(epoch)
                text = self.transcribe(audio)
                if not text or len(text) < 2:
                    continue
                print(f"  \033[33myou:\033[0m {text}", flush=True)
                reply = self.respond(text, epoch)
                print(f"  \033[32maegrys:\033[0m {reply}", flush=True)
                self.tracer.current.meta["text"] = text
                self.tracer.end_turn()
                # Phase 1 is half-duplex: don't listen while we're still talking.
                self.speaker.wait_drained()
        except KeyboardInterrupt:
            print("\n" + self.tracer.table(), flush=True)
        finally:
            self.stop()
