"""Speech, music and video (D-028), each from the command line, on the CPU.

  python tests/integration/test_media.py --kera build/bin/kera --models DIR [--skip-music]

DIR holds the small real models (downloaded if missing): Qwen3-TTS 1.7B (Q4_K_M, with its audio decoder, 1.5 GB) and
ACE-Step 1.5's smallest set (the 0.6B language model, the text encoder, the turbo diffusion model and the VAE,
3.3 GB). Checked:
- kera speak finds the model's audio decoder by itself and writes a WAV that has sound in it;
- kera music writes a 48 kHz stereo WAV of about the length asked for, with sound in it, and an MP3;
- kera serve --music DIR starts the music engine's server, with its page;
- kera video explains itself (a real video model is too big for this machine).
"""
import argparse
import array
import gzip
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request
import wave

SPEECH = ("https://huggingface.co/ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF/resolve/main/",
          ["Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf", "mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf"])
MUSIC = ("https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/resolve/main/",
         ["acestep-5Hz-lm-0.6B-Q8_0.gguf", "Qwen3-Embedding-0.6B-Q8_0.gguf", "acestep-v15-turbo-Q4_K_M.gguf",
          "vae-BF16.gguf"])


def fetch(folder, base, names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        if not (folder / n).is_file():
            print(f"downloading {n}")
            tmp = folder / (n + ".part")
            urllib.request.urlretrieve(base + n, tmp)
            tmp.rename(folder / n)


def wav_info(path):
    """(rate, channels, seconds, loudness as RMS of 16-bit samples)"""
    with wave.open(str(path)) as w:
        n = w.getnframes()
        a = array.array("h", w.readframes(n))
        step = max(1, len(a) // 200000)
        s = a[::step]
        rms = math.sqrt(sum(x * x for x in s) / max(1, len(s)))
        return w.getframerate(), w.getnchannels(), n / w.getframerate(), rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--models", required=True, type=pathlib.Path)
    ap.add_argument("--port", type=int, default=8690)
    ap.add_argument("--skip-music", action="store_true")
    a = ap.parse_args()
    a.kera = os.path.abspath(a.kera)
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-media-"))
    env = dict(os.environ, KERA_HOME=str(work / "home"))
    failures = []

    def check(ok, what, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what + (f": {detail}" if detail else ""))

    def run(args, timeout=1800):
        t = time.time()
        r = subprocess.run([a.kera] + args, capture_output=True, text=True, env=env, timeout=timeout, cwd=work)
        return r, time.time() - t

    # speech
    speech = a.models / "speech"
    fetch(speech, *SPEECH)
    out = work / "hello.wav"
    r, t = run(["speak", "-m", str(speech / SPEECH[1][0]), "-p", "Hello, this is Kera speaking.", "-o", str(out)])
    check(r.returncode == 0, f"kera speak ran ({t:.0f} s)", r.stderr[-2000:])
    check("audio decoder" in r.stderr, "and found the model's audio decoder by itself")
    if out.is_file():
        rate, ch, secs, rms = wav_info(out)
        check(rate == 24000 and 0.8 < secs < 10, f"a 24 kHz WAV of {secs:.1f} s")
        check(rms > 100, f"with sound in it (loudness {rms:.0f})")
    else:
        check(False, "kera speak wrote a WAV")

    # music
    if not a.skip_music:
        music = a.models / "music"
        fetch(music, *MUSIC)
        out = work / "song.wav"
        r, t = run(["music", "-p", "calm lo-fi piano with soft drums", "--lyrics", "[Instrumental]", "--duration", "8",
                    "--seed", "7", "--models", str(music), "-o", str(out)])
        check(r.returncode == 0, f"kera music ran ({t:.0f} s)", r.stderr[-3000:])
        if out.is_file():
            rate, ch, secs, rms = wav_info(out)
            check(rate == 48000 and ch == 2, f"a 48 kHz stereo WAV ({rate} Hz, {ch} channels)")
            check(5 < secs < 11, f"of about the length asked for (8 s): {secs:.1f} s")
            check(rms > 100, f"with sound in it (loudness {rms:.0f})")
        else:
            check(False, "kera music wrote a WAV")
        mp3 = work / "song.mp3"
        r, t = run(["music", "-p", "calm lo-fi piano", "--lyrics", "[Instrumental]", "--duration", "5", "--seed", "1",
                    "--models", str(music), "-o", str(mp3)])
        data = mp3.read_bytes() if mp3.is_file() else b""
        check(r.returncode == 0 and len(data) > 10000 and (data[:3] == b"ID3" or data[0] == 0xFF),
              f"and an MP3 ({len(data)} bytes, {t:.0f} s)", r.stderr[-2000:])

        # the music engine's server and its page
        log = work / "music-serve.log"
        p = subprocess.Popen([a.kera, "serve", "--music", str(music), "--port", str(a.port)], stdout=open(log, "w"),
                             stderr=subprocess.STDOUT, env=env)
        try:
            page = ""
            for _ in range(240):
                if p.poll() is not None:
                    break
                try:
                    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/", headers={"Accept-Encoding": "gzip"})
                    with urllib.request.urlopen(req, timeout=2) as resp:
                        body = resp.read()
                        if resp.headers.get("Content-Encoding") == "gzip" or body[:2] == b"\x1f\x8b":
                            body = gzip.decompress(body)
                        page = body.decode(errors="replace")
                    break
                except Exception:
                    time.sleep(0.5)
            check("<html" in page.lower(), "kera serve --music serves the music engine's page", log.read_text()[-2000:])
        finally:
            p.terminate()
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()

    # video: the plumbing (the smallest real video model is several GB)
    r, _ = run(["video", "--help"], timeout=60)
    check(r.returncode == 0 and "kera video" in r.stdout and "Wan" in r.stdout, "kera video explains itself",
          r.stdout[:500])

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
