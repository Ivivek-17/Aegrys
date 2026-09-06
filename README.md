# Aegrys

A fully local voice assistant: streaming speech-to-text, local LLM reasoning,
MCP tool execution, local text-to-speech, and barge-in — running entirely on a
**15 W laptop CPU with no GPU**.

No API keys. No network calls for inference. Everything runs on-device.

```
mic ─► ring buffer ─► Silero VAD ─► endpointer ─► faster-whisper ─┐
        48kHz→16kHz      32ms frames    adaptive silence          │
                                                                  ▼
                        ┌──────────── turn orchestrator (epoch-tagged) ────────────┐
                        │                                                          │
                        │   constrained router ──► MCP tools (4 stdio subprocs)    │
                        │    (~120 tok prompt)      timers · reminders             │
                        │                           calendar · email               │
                        │                                    │                     │
                        │   streaming LLM ◄──── tool result (untrusted, no tools)  │
                        └──────────────────────┬───────────────────────────────────┘
                                               ▼
                              clause chunker ─► Piper TTS ─► 30ms output chunks
                                               │
     barge-in: sustained speech during playback bumps the epoch,
     flushes the queue, drops the LLM stream ──┘
```

---

## Measured performance

All numbers from an **Intel i7-1355U** (2 P-cores + 8 E-cores, 15 W), Iris Xe
(no CUDA), 16 GB RAM, Windows 11. Reproduce with `bench/`.

### Component latency

| Stage | Model | Measured |
|---|---|---|
| VAD | Silero v5 ONNX | **0.11 ms** per 32 ms frame (295× headroom) |
| STT | faster-whisper `tiny.en` int8 | **355–490 ms**, **4.34% WER** |
| LLM | Qwen2.5-3B-Instruct Q4_K_M | **12.7 tok/s** decode |
| TTS | Piper `en_US-amy-low` | **111 ms** → 2.13 s audio (**RTF 0.05**) |

### End-to-end

| Path | TTFA (median) | TTFA (max) |
|---|---|---|
| Tool turn (router mode) | **1999 ms** | 2968 ms |
| Tool turn (native binding) | 2985 ms | **23630 ms** |
| Conversational turn | ~1700–2500 ms | — |

TTFA = *speech-end to first audio sample out*. Not time-to-first-token — the user
can't hear tokens. Voice input adds ~400 ms endpointing + ~490 ms STT on top of
the text-mode figures above.

### Speech recognition accuracy

120 LibriSpeech test-clean utterances, 2671 words:

| Model | WER | Decode |
|---|---|---|
| **`tiny.en`** (chosen) | **4.34%** | 1× |
| `base.en` | 3.56% | ~2× |
| `small.en` | 2.40% | ~2.5× |

`tiny.en`'s errors are almost entirely proper nouns from read audiobook prose
("Stephanos Dedalos" → "stefanos dead loss") — a class that doesn't occur in voice
commands. Paying 2–3× decode time for 0.78 WER points on vocabulary we never hear
is a bad trade here.

### Voice path (closed-loop, synthetic speech)

| Utterance | Detected | Transcribed | Routed |
|---|---|---|---|
| "set a timer for five minutes" | ✅ | "Set a timer for 5 minutes." | `set_timer` |
| "what is on my calendar today" | ✅ | exact | `list_events` |
| "remind me to call mom tomorrow" | ✅ | exact | `add_reminder` |
| "what is the capital of France" | ✅ | exact | `respond` |

4/4 detected and routed; endpoint fires 40–206 ms after speech ends.

### Barge-in

| Property | Measured | Bar |
|---|---|---|
| Audio queue flush | **0.24 ms** | — |
| Worst audible tail | **30.2 ms** | < 100 ms |
| LLM tokens leaked after interrupt | **0** | ≤ 2 |

### Under concurrent load

| Scenario | LLM | TTS RTF | VAD jitter |
|---|---|---|---|
| LLM + TTS | −4.7% | 0.138 | — |
| LLM + TTS + VAD | −7.1% | 0.167 | +1.25 ms |
| LLM + TTS + STT + VAD | −12.5% | **0.195** | **+2.10 ms** |

TTS stays 5× under the RTF 1.0 underrun threshold and VAD stays 6× under its
32 ms frame budget, so audio never stutters and barge-in stays responsive.

---

## Eight measurements that changed the design

The interesting part of this project is that most of the "obvious" choices were
wrong on this hardware, and only measurement revealed it. Full data in
[`bench/RESULTS.md`](bench/RESULTS.md).

**1. Whisper's cost is constant, not proportional to utterance length.**
3 s → 2641 ms, 10 s → 2795 ms. Whisper pads every input to a fixed 30-second mel
window, so cost is *encoder*-dominated. Short utterances are not cheap.

**2. So `distil-whisper` is the wrong family for CPU.** Distillation shrinks the
*decoder*; the cost here is the *encoder*, which `distil-small.en` inherits from
`small`. It measured **3.5× slower than plain `base.en`**. We use `tiny.en`.

**3. Streaming STT partials aren't affordable.** Each partial costs a *full*
encoder pass. Re-decoding every 500 ms would burn a 66% duty cycle on 6 threads —
stolen directly from the LLM, which is the real bottleneck. We endpoint, then decode.

**4. Kokoro-82M can't sustain real-time here; int8 made it worse.** Kokoro fp32
best case was **RTF 1.055** — it can only just keep up with playback while using
the same 6 threads the LLM needs. And its int8 build was **3–4× slower than fp32**
(quantized ops falling off the optimized kernel path). Piper is **RTF 0.05–0.15**.
Never assume int8 is faster on CPU.

**5. More threads are slower.** On this hybrid P/E-core chip the LLM peaks at
`num_thread=6`; 12 threads is **19% worse**. Also: ollama silently **reloads the
model whenever `num_thread` changes between requests**, so it cannot be swept
per-request.

**6. Binding 10 MCP tools natively cost 20 seconds.** The tool schema makes the
prompt ~900 tokens, and a cold prefill of 900 tokens runs at ~50 tok/s on this CPU —
**15–20 s**. Two small schema-constrained calls (~270 tokens total) are **1.5×
faster on median TTFA and 8× better on the tail**, at identical routing accuracy.
On a GPU, where 900 tokens of prefill is ~100 ms, native binding would win — this is
a hardware-specific conclusion, not a universal one.

**7. Thermal throttling is real — the early soak tests were too short.** One- and
ten-minute runs showed no drift, so it was written off. Over **50 minutes** of
sustained load the same work runs **1.8× slower** (392 ms → 695 ms), and recovers
fully after **4 minutes idle**. It's a 15 W chip. Treat any benchmark longer than
~10 minutes as thermally contaminated.

**8. Silero VAD fails silently without a context buffer.** v5 needs **64 samples
from the previous chunk prepended** to each 512-sample frame. Feed it a bare 512
and it doesn't raise — it returns ~0.001 for clear human speech. Voice input
detected *nothing*, and unit-testing the wrapper in isolation would never have
caught it; only running real audio through the real path did.

| Input | Max prob | Speech frames |
|---|---|---|
| 512, no context | 0.004 | 0/343 |
| 64 context + 512 | **1.000** | **233/343** |

**Bonus: an easy eval set lies.** The first routing eval reported 8/8 — then a live
run routed *"thanks that is all"* to `add_reminder`. Every case in that set was
either a clean tool request or obvious chit-chat. With only tools described in
detail, the model treats every utterance as a tool request and picks the nearest
match. Naming `respond` as the explicit default and adding the hard cases —
closings, acknowledgements, timer-vs-reminder near-misses — took it to
**22/22 (100%)**.

---

## Design decisions worth explaining

**Barge-in is a cancellation problem, not a feature flag.** Every turn gets a
monotonically increasing epoch; work carries the epoch it started under, and every
stage boundary drops stale results. A boolean stop-flag can't express "abandon work
already in flight" — there's always a race between checking the flag and emitting.

**Never hand the audio device more than 30 ms.** This is where naive barge-in
implementations fail: you cancel generation and the speaker keeps talking because
the OS already holds two seconds of audio. Playback is fed from a flushable deque,
which bounds the audible tail at one callback period — measured at 30.2 ms.

**Prompt injection is handled structurally, not with a plea.** Tool output —
especially email — is untrusted input. The call that *sees* tool output has **no
tools bound**, so it is incapable of acting on anything the content instructs. The
call that *can* act never sees the content. `tests/test_injection.py` asserts this
end to end against a hostile mailbox fixture.

**LangChain is deliberately not in the hot path.** `AgentExecutor` can't express
mid-generation cancellation or streaming partial output into TTS — which is the
entire product. The turn loop is ~200 hand-written lines where cancellation is
explicit and readable.

**WASAPI does not resample.** Opening the mic at 16 kHz raises `Invalid sample
rate`. Devices are natively 48 kHz, so capture happens at 48 k and converts in
software via a 3:1 polyphase decimator.

---

## Running it

Requires Python 3.12+ and [ollama](https://ollama.com).

```bash
uv venv && uv pip install -e .
```

Fetch models (~2.4 GB total):

```bash
ollama pull qwen2.5:3b-instruct-q4_K_M
```

Piper voice, Silero VAD, and the Whisper model download on first use. Then seed the
local calendar and mailbox fixtures:

```bash
python scripts/seed_demo_data.py
```

Start the LLM backend:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_MODELS=D:/Aegrys/.cache/ollama ollama serve
```

Talk to it:

```bash
python -m aegrys.cli
```

Or without a microphone:

```bash
python -m aegrys.cli --text
```

Useful flags: `--devices` (list audio devices), `--say "..."` (one turn),
`--no-tools`, `--no-barge-in`, `--stt-model base.en`, `--trace-file traces.jsonl`.

**Use headphones**, or see the caveat below.

---

## Tools

Four MCP servers run as stdio subprocesses, exposing 10 tools:

| Server | Tools | Backing |
|---|---|---|
| timers | `set_timer` `list_timers` `cancel_timer` | SQLite |
| reminders | `add_reminder` `list_reminders` `complete_reminder` | SQLite |
| calendar | `list_events` `next_event` | local `.ics` |
| email | `summarize_email` `count_email` | local `.eml` |

Calendar is a local iCalendar file and email reads a local maildir, so the
"fully local" claim holds. **Real IMAP would be a network call** — the inference
would still be local, but the data would not. Timers actually fire and announce
themselves.

---

## Testing

```bash
python -m pytest tests/ -q
```

18 unit tests covering epoch cancellation, audio buffer bounds, clause chunking,
adaptive endpointing, and the echo guard. `tests/test_injection.py` additionally
requires a running LLM backend.

`bench/` holds the twelve measurement harnesses (S1–S12) that produced every number
above, plus raw JSON in `bench/results/`.

---

## Known limitations

**Echo is unresolved (R1).** On laptop speakers plus a laptop mic, the assistant may
hear itself and barge in on its own voice. The baseline mitigation is a half-duplex
energy gate requiring 300 ms of sustained speech above the playback level. Automated
measurement was **inconclusive** — a muted speaker and perfect hardware AEC produce
identical data, and the loopback check that would distinguish them failed to open.
**Use headphones for a clean demo** until a human test settles it.

**The voice path is verified only against synthetic speech.** `bench/s11_voice_loop.py`
drives real audio through the actual mic code path (streaming decimator, framing,
VAD, endpointer), but the speech comes from our own TTS. Microphone gain, background
noise, accents, and real human voices are untested.

**Performance degrades ~1.8× under sustained load** and recovers after a few minutes
idle. Fine for conversational use; noticeable if you hammer it.

**Reminders don't fire.** They're stored with the user's own wording ("tomorrow at
6") rather than a parsed datetime, so they can be listed but not scheduled. Timers do
fire.

**Single-turn tool use.** One tool call per turn; no chaining or multi-step plans.

**Reminders and calendar are read-mostly.** No recurring events, no editing.
