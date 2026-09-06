"""S3b — sanity-check S3's PASS.

S3 measured only +1.0 dB of echo leak and 0.059 correlation. That is either
(a) hardware AEC in the Intel Smart Sound mic array, or (b) nothing actually came
out of the speakers. Those look identical in the data, so distinguish them:

  - capture the render path via "Stereo Mix" loopback while playing a loud tone.
    Signal there  => audio really was emitted, and (a) is the explanation.
    No signal     => (b), and S3's PASS is meaningless.
"""

import numpy as np
import sounddevice as sd

DUR = 2.0


def db(x):
    return 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-10))


def find(name_sub, hostapi_sub, want_input):
    hostapis = [a["name"] for a in sd.query_hostapis()]
    for i, d in enumerate(sd.query_devices()):
        ch = d["max_input_channels"] if want_input else d["max_output_channels"]
        if (ch > 0 and name_sub.lower() in d["name"].lower()
                and hostapi_sub.lower() in hostapis[d["hostapi"]].lower()):
            return i, d["name"]
    return None, None


def main():
    out_i, out_n = find("Speakers", "WASAPI", False)
    mix_i, mix_n = find("Stereo Mix", "WDM", True)
    mic_i, mic_n = find("Microphone Array", "WASAPI", True)
    print(f"speakers   : [{out_i}] {out_n}")
    print(f"loopback   : [{mix_i}] {mix_n}")
    print(f"mic array  : [{mic_i}] {mic_n}")

    sr = int(sd.query_devices(out_i)["default_samplerate"])
    t = np.arange(int(DUR * sr)) / sr
    # loud, unmistakable 440 Hz tone at -6 dBFS
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    print(f"\nplaying a {DUR}s 440 Hz tone at -6 dBFS ({sr} Hz)...")

    if mix_i is not None:
        mix_sr = int(sd.query_devices(mix_i)["default_samplerate"])
        try:
            rec = sd.playrec(tone.reshape(-1, 1), samplerate=sr, channels=1,
                             device=(mix_i, out_i), dtype="float32")
            sd.wait()
            lvl = db(rec[:, 0])
            print(f"  loopback (Stereo Mix) level during tone: {lvl:6.1f} dBFS")
            if lvl > -50:
                print("  => render path CONFIRMED: audio really was emitted.")
                render_ok = True
            else:
                print("  => render path SILENT: nothing was actually played.")
                render_ok = False
        except Exception as e:
            print(f"  loopback capture failed: {type(e).__name__}: {e}")
            render_ok = None
    else:
        render_ok = None
        print("  no Stereo Mix device found")

    # and what does the mic hear from that same loud tone?
    print("\nnow measuring what the MIC hears from the same tone...")
    silence = sd.rec(int(1.0 * 48000), samplerate=48000, channels=1,
                     device=mic_i, dtype="float32")
    sd.wait()
    base = db(silence[:, 0])

    rec = sd.playrec(tone.reshape(-1, 1), samplerate=sr, channels=1,
                     device=(mic_i, out_i), dtype="float32")
    sd.wait()
    heard = db(rec[:, 0])

    # tone-specific energy: how much 440 Hz is in the mic signal?
    x = rec[:, 0]
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    k = int(np.argmin(np.abs(freqs - 440)))
    band = spec[max(0, k - 3):k + 4].max()
    total = spec.mean()
    snr = 20 * np.log10(max(band / max(total, 1e-12), 1e-12))

    print(f"  mic noise floor        : {base:6.1f} dBFS")
    print(f"  mic during tone        : {heard:6.1f} dBFS  ({heard-base:+.1f} dB)")
    print(f"  440 Hz peak vs mean bin: {snr:6.1f} dB")

    print("\nverdict:")
    if render_ok is None:
        print("  INCONCLUSIVE: could not confirm the render path (loopback")
        print("  capture unavailable - PortAudio refuses to mix a WDM input")
        print("  with a WASAPI output). A silent speaker and perfect AEC are")
        print("  INDISTINGUISHABLE in this data, so S3's PASS is NOT earned.")
        print("  => R1 stays OPEN. Needs a 30-second human test:")
        print("     play audio, confirm it is audible, speak over it.")
    elif render_ok is False:
        print("  S3 PASS WAS SPURIOUS - nothing was playing. Re-test needed.")
    elif snr > 20:
        print("  Mic clearly picks up the tone -> real acoustic coupling exists.")
        print("  S3's low leak on speech may be AEC tuned for voice; treat R1")
        print("  as OPEN and confirm with a human speaking during playback.")
    else:
        print("  Render path worked, yet the mic does NOT pick up the tone.")
        print("  Consistent with hardware AEC / echo suppression on the mic array.")
        print("  R1 looks genuinely mitigated, but confirm with a human test.")


if __name__ == "__main__":
    main()
