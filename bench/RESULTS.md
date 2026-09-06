# Phase 0 Spike Results

**Machine:** i7-1355U (2 P + 8 E, 12 threads, 15W), Iris Xe (no CUDA), 16 GB RAM, Windows 11
**Date:** 2026-09-06 · Raw JSON in `bench/results/`, scripts in `bench/`

| Spike | Bar | Result | Verdict |
|---|---|---|---|
| S1 TTS | Kokoro RTF < 0.6 | Kokoro best **RTF 1.055**; Piper **RTF 0.15** | **FAIL → swap to Piper** |
| S2 LLM | ≥ 8 tok/s sustained | **12.66 tok/s**, no thermal drift, routing 10/10 | **PASS** |
| S3 Audio | echo gate viable | full-duplex OK; echo measurement **inconclusive** | **OPEN** |
| S4 STT | 3s decode < 400 ms | `tiny.en` **294 ms**; `base.en` 753 ms | **PASS (different model)** |
| S5 Concurrency | RTF < 1.0, VAD jitter << 32 ms | RTF **0.195**, jitter **5.3 ms**, LLM −12.5% | **PASS** |

---

## Environment blocker found first

**C: was 100% full — 0 bytes free** (909 MB even after cleanup). Every model download failed
until caches were redirected to D:. Unrelated to this project but it will break other things;
worth clearing independently. All caches now live under `D:/Aegrys/.cache/` via `bench/env.sh`.

Also: `ollama pull` **exits 0 on a disk-full failure**. Don't trust its exit code — check `ollama list`.

---

## S4 — STT: cost is constant, and distil is the wrong family for CPU

Decode time barely moves with utterance length (`distil-small.en`: 3s→2641 ms, 5s→2691 ms,
10s→2795 ms). Whisper pads every input to a **fixed 30-second mel window**, so cost is
**encoder-dominated and essentially constant**.

Two consequences:

1. **`distil-*` models do not help on CPU.** They shrink the *decoder*; the cost here is the
   *encoder*, which `distil-small.en` inherits from `small`. It measured **3.5× slower than
   plain `base.en`** (2641 ms vs 739 ms). The original design doc recommended `distil-small.en`
   — that was wrong.
2. **`chunk_length` is not a lever.** Shortening the window made it *slower*
   (base.en: 10s window → 1879 ms vs 1127 ms default). The 30s window is effectively fixed.

### Model × threads (3s clip, int8, beam 1)

| Model | t=2 | t=4 | t=6 | t=8 |
|---|---|---|---|---|
| `tiny.en` | 487 ms | 392 ms | **294 ms** | 353 ms |
| `base.en` | 1236 ms | 1069 ms | 890 ms | 753 ms |
| `small.en` | — | — | ~2900 ms | — |

Both `tiny.en` and `base.en` scored **0% WER** on the full 11s JFK clip. `tiny.en`'s apparent
error in the first sweep came from a 3s slice cutting mid-word — a fixture artifact, not the model.

> **Accuracy is NOT settled.** One clip is not an evaluation. Before locking `tiny.en`, run a real
> WER comparison on ~100 LibriSpeech test-clean utterances plus in-domain command phrases.

### The consequence that changes the architecture

**Streaming partials are not viable.** Each partial costs a *full* encoder pass (~330 ms on
`tiny.en`). Partials every 500 ms = a **66% duty cycle on 6 threads, continuously**, taken
directly from the LLM — which S2 shows is the real bottleneck. On `base.en` it's outright
impossible: a partial takes longer than the interval.

→ Use **endpoint-then-decode**. If a live transcript is wanted for the demo, render it *after*
the single decode, not from continuous re-decoding.

---

## S2 — LLM: passes comfortably, but the router design is too slow

`qwen2.5:3b-instruct-q4_K_M` via ollama (own server on :11435, models on D:).

**Thread sweep confirms the §3 hybrid-core hypothesis:**

| num_thread | 2 | 4 | **6** | 8 | 12 |
|---|---|---|---|---|---|
| decode tok/s | 9.33 | 11.34 | **12.66** | 11.04 | 10.21 |

**More threads are worse.** 12 threads is 19% slower than 6 — the E-cores drag the batch.
Prefill is fast (~258 tok/s) and is *not* a bottleneck.

**No thermal throttling:** 6 consecutive runs drifted +4.4% (12.89 → 12.32 tok/s). R5 is
lower-risk than assumed — though this was a ~1-minute burst, not a real soak test.

### Schema-constrained routing works

**8/8 valid JSON**, zero malformed output. §6.1's claim holds: constrained decoding makes a 3B
model a reliable router. Sharpening the tool descriptions took accuracy from 7/10 → **10/10**
(the original miss: "remind me to call mom tomorrow at 6" → `set_timer`, fixed by explicitly
contrasting *duration* vs *clock time*).

### The problem: tokens, not calls

At 12.66 tok/s, **every emitted token costs 79 ms.** Router latency is dominated by output length:

| Router encoding | tokens | latency | accuracy |
|---|---|---|---|
| `{tool, args}` full JSON | 22.5 | 2060 ms | 9/10 |
| `{tool}` only | 10.0 | 1301 ms | 9/10 |
| `{tool}` + sharpened prompt | 9.0 | **1096 ms** | **10/10** |

Dropping `args` from the router schema cut latency 47%. But:

```
router 1096 ms + first sentence 1279 ms = 2375 ms of LLM alone
```

**The two-call router→synthesis design in §6.2 cannot hit the TTFA target.** Streaming TTFT
measured 513–1057 ms; time-to-first-sentence 587–1500 ms.

→ **Invert it: single streaming call**, detect a tool call on the token stream. Conversational
turns (the common case) then pay TTFT only and skip the router entirely. Tool turns pay the
second call, masked by filler audio.

---

## S1 — TTS: Kokoro fails on this CPU, Piper passes

| Model | config | short sentence | RTF |
|---|---|---|---|
| Kokoro int8 | intra=6 | 6702 ms | 3.570 |
| Kokoro int8 | intra=2 | 10185 ms | 5.425 |
| Kokoro fp32 | intra=6 | 2003 ms | **1.055** |
| **Piper `en_US-amy-low`** | 4 threads | **393 ms** | **0.151** |
| Piper `en_US-lessac-medium` | 4 threads | 413 ms | 0.217 |

Two findings:

1. **int8 is 3–4× SLOWER than fp32.** Dynamic ONNX quantization is a *pessimization* here —
   the quantized ops fall off the optimized kernel path. Never assume int8 is faster; measure.
2. **Kokoro at RTF ~1.05 has zero headroom.** It can only just keep up with playback while
   using 6 threads — the same 6 the LLM needs. Under real concurrency it exceeds RTF 1.0 and
   the audio underruns. Not viable.

**Piper is 7× faster and passes with margin.** R2's fallback is now the primary.

Cost: Piper's voice quality is below Kokoro's. For a portfolio demo that trades well against
an assistant that stutters — but listen to `bench/piper_*.wav` and confirm it's acceptable.

**Windows gotcha:** espeak-ng *does* work (`espeakng-loader` bundles the DLL — R2's install risk
was overstated), but it returns IPA and the default cp1252 console **crashes** on characters like
`ɛ`. `PYTHONIOENCODING=utf-8` is mandatory.

---

## S3 — Audio: full-duplex works, echo unresolved

**Confirmed:**
- Full-duplex play+record opens fine on WASAPI.
- **WASAPI does not resample.** Opening the mic at 16 kHz raises `Invalid sample rate
  [PaErrorCode -9997]`. Devices are natively **48 kHz** — capture at 48k and downsample to 16k
  in software for Silero/Whisper. This is a hard implementation requirement.
- PortAudio refuses to mix a WDM input with a WASAPI output (`-9993`), which blocked the
  loopback verification.

**Not confirmed — R1 stays OPEN.** The measurement showed only +1.0 dB of echo leak and 0.059
correlation, which *looks* like hardware AEC on the Intel Smart Sound mic array. But the
loopback check that would prove audio was actually emitted failed to open. **A silent speaker
and perfect AEC produce identical data.** The PASS is not earned.

> **Needs a 30-second human test:** play `bench/piper_en_US-lessac-medium.wav` through the
> laptop speakers, confirm it is audible, then speak over it and check whether the mic
> transcribes the assistant's own voice.

---

## S5 — Concurrency: the thread budget holds

Open item #3 from the first pass. Every earlier number was measured with one component
running; the real pipeline overlaps them by design (§3).

**Two failed attempts before a valid one** — both worth recording, because each was a real
measurement artifact:

1. **S5 (invalid).** The "solo" baseline ran cold right after a server restart (TTFT 7.8 s =
   model load), so a *contended* run measured **92% faster than solo**. Any benchmark where
   contention improves throughput is measuring something else.
2. **S5b (invalid).** Fixed the warm-up, but 7/10 scenarios still reloaded the model.
   Cause: **ollama reloads the model whenever `num_thread` changes between requests.**
   Per-request LLM thread sweeping is impossible — set it at server level or accept a reload.
   This also confounded a −20% baseline drift with reload cost.
3. **S5c (valid).** `num_thread` pinned at 6 → **0 reloads**. A solo baseline runs *before and
   after every scenario*, so each is scored against its local bracketing baseline and thermal
   drift cancels.

### Results (LLM pinned at `num_thread=6`)

| Scenario | LLM tok/s | local base | degradation | TTS RTF | VAD jitter p99 |
|---|---|---|---|---|---|
| LLM + TTS t=2 | 13.21 | 13.86 | **−4.7%** | 0.138 | — |
| LLM + TTS t=4 | 12.67 | 13.32 | **−4.9%** | 0.126 | — |
| LLM + TTS t=2 + VAD | 13.04 | 14.03 | **−7.1%** | 0.167 | +1.25 ms |
| LLM + TTS t=2 + STT t=2 + VAD | 12.27 | 14.03 | **−12.5%** | 0.195 | +2.10 ms |

Baseline trace: 15.03 → 12.68 → 13.96 → 14.10 → 13.96 tok/s. The opening 15.03 is a warm
outlier; it then settles on a ~13.9 plateau and **stops declining**. That is warm-up, not
runaway throttling — R5 stays low-risk even under ~10 minutes of sustained load.

### The three properties that matter

1. **TTS RTF worst case 0.223 (p95) — PASS with 4.5× margin.** Piper never approaches the
   RTF 1.0 underrun threshold under any load. This is the single most important result: it
   confirms the audio output will not stutter, which is exactly what Kokoro (RTF 1.055 *solo*)
   would have failed.
2. **VAD jitter max +5.3 ms against a 32 ms frame budget — PASS with 6× margin.** Inference
   is 1.8–2.9 ms p99. Barge-in detection stays real-time even at full load, which validates
   giving VAD its own dedicated thread (§3).
3. **LLM degradation 4.7–12.5%** — bounded and monotonic with load.

### Two secondary findings

- **Give Piper 4 threads, not 2.** t=4 costs the same LLM degradation as t=2 (−4.9% vs −4.7%,
  within noise) but yields better RTF (0.126 vs 0.138). §3's allocation of 2 is too stingy.
- **STT degrades most: 1305 ms under full load** vs 487 ms solo at t=2 (2.7×). But this
  scenario is *pessimistic by construction* — it holds the LLM generating while STT runs. In a
  real barge-in the LLM is cancelled first, returning its 6 threads. **Cancellation is also a
  performance optimization, not just a UX feature.** Worth stating explicitly in the design.

### Effect on the latency budget

Under realistic response-phase load (LLM + TTS + VAD, −7.1%), LLM-to-first-sentence moves
~1280 → ~1375 ms and Piper's first sentence ~400 → ~435 ms. **Contention costs roughly 130 ms
of TTFA — it is not a significant factor.** The thread budget in §3 works essentially as designed.

---

## S6-S9 — build-phase measurements

Run during Phases 2-4. Scripts: `s6_bargein.py`, `s7_tool_latency.py`,
`s8_cache_thrash.py`, `s9_tool_modes.py`.

### S6 — barge-in latency (Phase 2 exit criterion: <100 ms)

| Property | Measured | Bar | |
|---|---|---|---|
| Audio queue flush | 0.24 ms | — | |
| Worst audible tail | 30.2 ms | < 100 ms | **PASS** |
| LLM tokens after interrupt | 0 | <= 2 | **PASS** |

Over 2 s of queued audio dropped instantly. The tail is bounded by `out_chunk_ms`,
which validates §4.2's rule about never over-filling the device buffer.

### S7/S8 — why tool turns cost 18-22 s

Phase 3 shipped native tool binding and live TTFA was 18240 ms and 21978 ms.

**First hypothesis was wrong.** I assumed the two calls of a tool turn (tools-bound,
then tool-result) were evicting each other from ollama's prompt cache. S8 tested
alternating prefixes directly: prefill stayed 62-75 ms and a full tool turn cost
2862 ms. Not the cause.

Instrumenting the live app with ollama's own timings found it:

| Turn | prompt tokens | prefill |
|---|---|---|
| 1 | 861 | 91 ms |
| 2 | 886 | **15837 ms** |
| 3 | 912 | **20233 ms** |

The tool schema makes the prompt ~900 tokens, and when the prefix cache misses,
prefill runs at **~50 tok/s** on this CPU. Note S2 measured prefill at 258 tok/s on
short prompts -- prefill throughput does not hold at length.

### S9 — router vs native tool binding

| Mode | TTFA median | TTFA max | wall median | routing |
|---|---|---|---|---|
| **router** (2 small constrained calls, ~270 tok) | **1999 ms** | **2968 ms** | 4821 ms | 8/8 |
| native (10 schemas bound, ~900 tok) | 2985 ms | **23630 ms** | 7664 ms | 8/8 |

Identical accuracy. The argument for the router is not the 1.5x median -- it's the
**8x better tail**. Native latency is unpredictable, which is worse than merely slow.

> **Hardware-specific.** On a GPU, 900 tokens of prefill is ~100 ms and native
> binding would win. The router is right *here*, not universally.

### S10 — routing accuracy on a harder eval set

S9 reported 8/8, but a live run then routed **"thanks that is all" -> add_reminder**.
The eval set was too easy: every case was either a clean tool request or obvious
chit-chat. With only tools described in detail, the model treats every utterance as
a tool request and picks the nearest match.

Two fixes, both in prompt/description text rather than code:
- make `respond` the explicitly-stated DEFAULT and name the conversational cases
  ("thanks", "that's all", "never mind", "ok", "bye") outright;
- disambiguate `count_email` (how many) from `summarize_email` (read them).

| Set | Before | After |
|---|---|---|
| tool requests | 11/12 | **12/12** |
| conversational | 9/10 | **10/10** |
| **overall** | 20/22 | **22/22 (100%)** |

Router `pick()` latency is ~800-1200 ms warm. Lesson: an eval set of only clean
cases will report a number that does not survive contact with real speech.

### S11 — closed-loop voice test, and a silent VAD bug

Every number up to this point came from the TEXT path. The microphone path had
never been exercised. S11 closes the loop without a human: Piper synthesizes an
utterance, it is upsampled to 48 kHz and pushed through the **real** `MicStream`
queue, so the actual streaming decimator, 512-sample framing, Silero VAD and
adaptive endpointer all run exactly as they do live. Only the acoustic path is
substituted.

**It immediately found a bug that made voice input completely non-functional.**

Silero v5 requires **64 samples of context from the previous chunk prepended** to
each 512-sample frame — the model consumes 576 samples per step. Feeding a bare
512 does not raise; it silently returns ~0.001 for clear human speech.

| Input | max prob | speech frames (bench/jfk.wav) |
|---|---|---|
| 512, no context (original) | **0.004** | **0 / 343** |
| 64 context + 512 | **1.000** | **233 / 343** |

> **This invalidates the functional VAD claims in S5c.** Those runs timed a model
> that was executing but producing garbage. The *timing* numbers remain roughly
> valid (the compute is nearly identical — 576 vs 512 samples), but "VAD works
> under load" was never actually demonstrated until now. Re-measured post-fix
> figures are below.
>
> Guarded by `tests/test_core.py::test_vad_detects_real_speech`, which fails if
> the context buffer is ever removed.

**Post-fix, the voice path works end to end:**

| Utterance | detected | STT | routed |
|---|---|---|---|
| "set a timer for five minutes" | yes | "Set a timer for 5 minutes." | `set_timer` |
| "what is on my calendar today" | yes | exact | `list_events` |
| "remind me to call mom tomorrow" | yes | exact | `add_reminder` |
| "what is the capital of France" | yes | exact | `respond` |

4/4 detected, 4/4 routed correctly, 3/4 word-exact (the miss is "5" vs "five",
a numeral formatting difference). Endpoint decision fired 40-206 ms after audio
ended; STT median 692 ms.

**What S11 does NOT prove:** R1 (echo), microphone gain and noise handling, or
accuracy on real human voices. Synthetic speech is easier than the real thing.

### S12 — real WER evaluation, and thermal throttling is real after all

Closes the open STT-accuracy item. Phase 0 picked `tiny.en` on **latency alone**;
the only accuracy evidence was 0% WER on a single 11-second clip.

**120 LibriSpeech test-clean utterances, 2671 reference words:**

| Model | WER | vs `tiny.en` |
|---|---|---|
| **`tiny.en`** | **4.34%** | — |
| `base.en` | 3.56% | −18% relative (0.78 pts) |
| `small.en` | 2.40% | −45% relative (1.94 pts) |

**`tiny.en` stays.** The errors are almost entirely proper nouns from read
audiobook prose ("Stephanos Dedalos" → "stefanos dead loss", "Brother Mac Ardle" →
"brother maccardo") — a class that essentially does not occur in voice commands.
S11 got 3/4 command utterances word-exact, and the one miss was "5" vs "five".
Paying 2-3x the decode time for 0.78 WER points on vocabulary we never hear is a
bad trade on this hardware. Revisit if the assistant ever needs dictation.

**Confirms S4's constant-cost finding** on independent audio — decode time is flat
against utterance duration:

| duration | 2.7s | 3.3s | 5.2s | 6.6s | 10.4s |
|---|---|---|---|---|---|
| decode | 838 ms | 803 ms | 854 ms | 854 ms | 867 ms |

#### Thermal throttling: R5 must be REOPENED

S2 saw no drift over ~1 minute; S5c saw a plateau over ~10 minutes and I concluded
"low risk". Over a **~50-minute** sustained run that conclusion breaks. Identical
work (3s jfk clip, `tiny.en`, 4 threads):

| Machine state | Decode |
|---|---|
| Phase 0, cold | 392 ms |
| immediately after 50 min sustained load | **695 ms (1.8x slower)** |
| after 4 minutes idle | **355 ms (fully recovered)** |

**Sustained load costs ~1.8x, and it recovers within 4 minutes of idle.** This is a
15 W chip; the earlier soak tests were simply too short to see it. Practical
consequence: warm the machine up before a demo but do not hammer it, and treat any
benchmark run longer than ~10 minutes as thermally contaminated.

> **Caveat on S12's own latency figures.** The medians in the table above (3435 ms
> for `tiny.en`) were recorded *during* that 50-minute run and are **not reliable**;
> a controlled re-measure immediately afterwards gives ~854 ms on the same model and
> comparable audio. Thermal throttling explains part of the gap but not all of it,
> and I could not isolate the rest. **The WER numbers are unaffected** — accuracy
> does not depend on how fast the machine is running.

VAD timing re-measured after the context fix: **0.109 ms median, 0.309 ms p99**
against a 32 ms budget (295x headroom), so S5c's timing conclusion survives even
though its functional claim did not.

---

## Revised TTFA budget (measured, not estimated)

| Stage | Measured | Source |
|---|---|---|
| Endpoint decision | 400 ms | design assumption (unchanged) |
| STT — `tiny.en` @ t=6 | 330 ms | S4b |
| LLM — router + first sentence | 2375 ms | S2b |
| TTS — Piper first sentence | 400 ms | S1d |
| **TTFA (two-call design)** | **~3.5 s** | ✗ vs 1.5 s target |

With the single-call redesign (no router on conversational turns):

| Stage | Measured |
|---|---|
| Endpoint | 400 ms |
| STT | 330 ms |
| LLM to first sentence | ~1280 ms |
| Piper first sentence | 400 ms |
| **TTFA** | **~2.4 s** |

Still above the original 1.5 s target. **The LLM is the bottleneck and it is not close.**
Remaining levers, in order of expected value:

1. **Chunk TTS on clauses, not sentences.** Piper at RTF 0.15 can start on a 4–5 word phrase;
   no need to wait for a full sentence. Could pull ~400–600 ms.
2. **Filler audio on tool turns** — masks the second call entirely.
3. **Smaller model.** Qwen2.5-1.5B would roughly double decode rate; measure the routing
   accuracy cost against the 10-case set.
4. Cap response length hard — the system prompt should enforce one short sentence.

**Honest revised target: ~1.8–2.0 s real TTFA, ~700 ms perceived.** The original 1.5 s figure
was written before any measurement and should be retired.

---

## What changed in the design

| § | Original | Revised | Why |
|---|---|---|---|
| 5.2 | `distil-small.en` | **`tiny.en`** (pending WER eval) | distil shrinks the decoder; CPU cost is the encoder |
| 5.2 | streaming partials, default on | **endpoint-then-decode** | each partial = a full encoder pass |
| 5.3 | speculative decode start | **drop** | constant-cost encoder makes it near-useless |
| 6.2 | two-call router → synthesis | **single streaming call** | router costs 1096 ms before synthesis starts |
| 6.3 | Kokoro-82M | **Piper** | Kokoro RTF 1.055 vs Piper 0.151 |
| 3 | llama.cpp `-t 4–6` | **`num_thread=6` confirmed** | measured; 12 threads is 19% worse |
| 7 | TTFA ~1.5 s | **~1.8–2.0 s** | measured |
| R2 | Kokoro G2P won't install | **install fine; perf is the problem** | risk was real, cause was wrong |
| R5 | thermal throttling | **low risk** (+4.4% over 6 runs) | measured |

## Open items

1. **R1 echo** — needs the human test above. Highest remaining unknown.
2. **STT accuracy** — `tiny.en` vs `base.en` WER on a real test set (~100 utterances).
3. ~~**Concurrency**~~ — **RESOLVED by S5c.** Thread budget holds: TTS RTF 0.195 (bar 1.0),
   VAD jitter 5.3 ms (bar 32 ms), LLM −12.5% worst case. Contention costs ~130 ms of TTFA.
4. **Piper voice quality** — subjective sign-off needed on the sample WAVs.
