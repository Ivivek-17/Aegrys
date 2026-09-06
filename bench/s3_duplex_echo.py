"""S3 — full-duplex audio and acoustic echo coupling (docs/DESIGN.md R1).

R1 is the top-ranked risk: laptop speakers + laptop mic means the assistant may
hear ITSELF and barge in on its own voice.

This plays a known TTS sample through the SPEAKERS while recording from the
built-in MIC ARRAY, then measures how much of the playback leaks into the mic.
Reports an echo-to-silence ratio in dB: the higher, the worse the coupling.

Pass criteria: a half-duplex energy gate can separate self-audio from real speech,
i.e. there is a usable margin between playback-leak level and real-speech level.

NOTE: this plays sound out loud. Uses laptop speakers + laptop mic deliberately,
since that is the worst-case demo configuration (headphones would trivially pass).
"""

import json
import time
import wave

import numpy as np
import sounddevice as sd

SR = 16000
WAV = "piper_en_US-lessac-medium.wav"


def find(name_sub, hostapi_sub, want_input):
    hostapis = [a["name"] for a in sd.query_hostapis()]
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if (ch > 0 and name_sub.lower() in d["name"].lower()
                and hostapi_sub.lower() in hostapis[d["hostapi"]].lower()):
            return i, d["name"], hostapis[d["hostapi"]]
    return None, None, None


def load_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    a = a.astype(np.float32) / 32768.0
    if sr != SR:  # crude linear resample; fine for an energy measurement
        n = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    return a.astype(np.float32)


def db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-10))


def main():
    out_i, out_n, out_h = find("Speakers", "WASAPI", want_input=False)
    in_i, in_n, in_h = find("Microphone Array", "WASAPI", want_input=True)
    print(f"output: [{out_i}] {out_n} ({out_h})")
    print(f"input : [{in_i}] {in_n} ({in_h})")
    if out_i is None or in_i is None:
        print("FAIL: could not find laptop speaker + mic array on WASAPI")
        return

    # WASAPI (shared mode via PortAudio) will NOT resample: opening a stream at
    # 16 kHz raises "Invalid sample rate". Capture at the device-native rate and
    # downsample to 16 kHz for Silero/Whisper. This is a real design constraint.
    global SR
    dev_sr = int(sd.query_devices(in_i)["default_samplerate"])
    out_sr = int(sd.query_devices(out_i)["default_samplerate"])
    print(f"native rates: mic {dev_sr} Hz, speakers {out_sr} Hz "
          f"(WASAPI does not resample)")
    SR = dev_sr

    results = {"output": out_n, "input": in_n, "mic_native_sr": dev_sr,
               "spk_native_sr": out_sr}

    # 1) baseline: room noise with nothing playing
    print("\n[1/3] recording 2s of silence (baseline room noise)...", flush=True)
    base = sd.rec(int(2 * SR), samplerate=SR, channels=1, device=in_i,
                  dtype="float32")
    sd.wait()
    base = base[:, 0]
    base_db = db(base)
    print(f"  baseline noise floor: {base_db:6.1f} dBFS")

    # 2) full-duplex: play the TTS sample while recording
    speech = load_wav(WAV)
    print(f"\n[2/3] full-duplex: playing {len(speech)/SR:.2f}s of TTS "
          f"through speakers while recording...", flush=True)
    rec = np.zeros((len(speech) + SR, 1), dtype=np.float32)
    try:
        t0 = time.perf_counter()
        recorded = sd.playrec(speech.reshape(-1, 1), samplerate=SR, channels=1,
                              device=(in_i, out_i), dtype="float32")
        sd.wait()
        dur = time.perf_counter() - t0
        rec = recorded
        duplex_ok = True
    except Exception as e:
        print(f"  FAIL full-duplex: {type(e).__name__}: {e}")
        results["duplex_error"] = str(e)
        duplex_ok = False

    if not duplex_ok:
        print("\nS3 FAIL: full-duplex stream could not be opened")
        with open("results/s3_duplex_echo.json", "w") as f:
            json.dump(results, f, indent=2)
        return

    echo = rec[:, 0]
    echo_db = db(echo)
    leak = echo_db - base_db
    print(f"  full-duplex opened OK, {dur:.2f}s wall")
    print(f"  mic level during playback: {echo_db:6.1f} dBFS")
    print(f"  echo leak above noise floor: {leak:+6.1f} dB")

    # cross-correlate to confirm what we captured really is the playback
    n = min(len(echo), len(speech), SR * 3)
    a = echo[:n] - echo[:n].mean()
    b = speech[:n] - speech[:n].mean()
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-12
    xc = np.correlate(a, b, mode="full") / denom
    lag = int(np.argmax(np.abs(xc)) - (n - 1))
    peak = float(np.max(np.abs(xc)))
    print(f"  peak correlation with played signal: {peak:.3f} "
          f"at lag {lag} samples ({lag/SR*1000:+.0f} ms)")

    results.update({"baseline_dbfs": round(base_db, 1),
                    "echo_dbfs": round(echo_db, 1),
                    "leak_db": round(leak, 1),
                    "xcorr_peak": round(peak, 3),
                    "lag_ms": round(lag / SR * 1000, 1)})

    # 3) verdict
    print("\n[3/3] verdict", flush=True)
    if leak < 6:
        print(f"  PASS: playback is only {leak:.1f} dB above the noise floor.")
        print("  The mic barely hears the speakers (likely hardware AEC / noise")
        print("  suppression in the Intel Smart Sound mic array).")
        print("  => a simple energy gate will separate self-audio from speech.")
        verdict = "PASS"
    elif leak < 20:
        print(f"  MARGINAL: {leak:.1f} dB of leak. An energy gate plus a")
        print("  sustained-speech requirement should work, but needs tuning.")
        verdict = "MARGINAL"
    else:
        print(f"  FAIL: {leak:.1f} dB of leak - the mic clearly hears the")
        print("  speakers. Half-duplex gating alone will not be enough;")
        print("  need real AEC or headphones for the demo.")
        verdict = "FAIL"
    results["verdict"] = verdict

    with open("results/s3_duplex_echo.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nS3 {verdict}")


if __name__ == "__main__":
    main()
