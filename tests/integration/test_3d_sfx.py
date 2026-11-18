"""3D models and sound effects (D-031), from the command line.

  python tests/integration/test_3d_sfx.py --kera build/bin/kera [--sfx-model FILE [--sfx-steps N]]
                                          [--3d-models DIR --picture FILE]

Without models (as in CI: the weights are 4 GB and more, and 3D needs a graphics chip to finish in reasonable time):
- kera 3d --help and kera sfx --help name the files to download and their licences, and kera help lists both;
- the engines' programs start from Kera's folder, and the sound-effect engine has its two models built in;
- a missing model folder, missing model files, a description without an image model and bad values are each
  explained, with exit code 2 (something to download) or 1 (a wrong option), before any engine runs.
With --sfx-model (MOSS-SoundEffect v2 or Stable Audio 3 Small SFX), a real sound: a WAV of the length asked for with
sound in it, the same file again with the same seed, and a different one with another seed. With --3d-models and
--picture, a real 3D model: a .glb file (binary glTF) that is the same again with the same seed.
"""
import argparse
import array
import hashlib
import math
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
import wave
import zlib


def png(path, w=64, h=64):
    """a small picture: a red square on white"""
    rows = b""
    for y in range(h):
        row = b"\x00"
        for x in range(w):
            inside = w // 4 <= x < 3 * w // 4 and h // 4 <= y < 3 * h // 4
            row += b"\xd0\x20\x20" if inside else b"\xff\xff\xff"
        rows += row

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def wav_info(path):
    """(rate, channels, seconds, loudness as RMS of 16-bit samples, or of float samples times 32767)"""
    with wave.open(str(path)) as w:
        n = w.getnframes()
        raw = w.readframes(n)
        rate, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
    if width == 2:
        s = array.array("h", raw)
    else:  # 32-bit: read as integers and scale (a float WAV can't be opened by the wave module anyway)
        s = [x / 65536 for x in array.array("i", raw)]
    step = max(1, len(s) // 200000)
    s = s[::step]
    rms = math.sqrt(sum(float(x) * float(x) for x in s) / max(1, len(s)))
    return rate, ch, n / rate, rms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--sfx-model", type=pathlib.Path, help="a sound-effect model, for a real run")
    ap.add_argument("--sfx-steps", type=int, default=0, help="fewer steps for a quicker real run (0: the model's)")
    ap.add_argument("--3d-models", dest="models3d", type=pathlib.Path, help="TRELLIS.2's files, for a real run")
    ap.add_argument("--picture", type=pathlib.Path, help="the picture for the real 3D run")
    a = ap.parse_args()
    a.kera = os.path.abspath(a.kera)
    kera_dir = pathlib.Path(a.kera).parent
    exe = ".exe" if os.name == "nt" else ""
    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-3d-sfx-"))
    env = dict(os.environ, KERA_HOME=str(work / "home"))
    failures = []

    def check(ok, what, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {what}")
        if not ok:
            failures.append(what + (f": {detail}" if detail else ""))

    def run(args, timeout=120):
        t = time.time()
        r = subprocess.run([a.kera] + args, capture_output=True, text=True, env=env, timeout=timeout, cwd=work)
        return r, time.time() - t

    # the commands explain themselves
    r, _ = run(["help"])
    check(r.returncode == 0 and "\n  3d " in r.stdout and "\n  sfx " in r.stdout, "kera help lists 3d and sfx", r.stdout)
    r, _ = run(["3d", "--help"])
    check(r.returncode == 0 and all(w in r.stdout for w in
                                    ["dinov3.gguf", "ss_flow.gguf", "shape_flow_1024.gguf", "tex_dec.gguf",
                                     "birefnet.gguf", "ilintar/trellis2-gguf", "DINOv3 License", "MIT"]),
          "kera 3d --help names the files, where they come from and their licences", r.stdout)
    r, _ = run(["sfx", "--help"])
    check(r.returncode == 0 and all(w in r.stdout for w in
                                    ["moss-soundeffect-v2-q8_0.gguf", "audio.cpp-gguf", "Apache-2.0",
                                     "Stability AI Community License"]),
          "kera sfx --help names the file, where it comes from and its licence", r.stdout)

    # the engines' programs start from Kera's folder
    trellis = kera_dir / "3d" / ("trellis-cli" + exe)
    audiocpp = kera_dir / "sfx" / ("audiocpp_cli" + exe)
    has3d, hassfx = trellis.is_file(), audiocpp.is_file()
    if has3d:
        r = subprocess.run([str(trellis), "--help"], capture_output=True, text=True, timeout=60)
        check(r.returncode == 0 and "usage:" in r.stderr, "the 3D engine starts", r.stderr[-2000:])
    else:
        print("skip the 3D engine's own checks: this build has no 3d/ folder")
    if hassfx:
        r = subprocess.run([str(audiocpp), "--list-loaders"], capture_output=True, text=True, timeout=60)
        check(r.returncode == 0 and "moss_soundeffect: gen" in r.stdout and "stable_audio: gen" in r.stdout,
              "the sound-effect engine starts, with MOSS-SoundEffect and Stable Audio", r.stdout + r.stderr[-2000:])
    else:
        print("skip the sound-effect engine's own checks: this build has no sfx/ folder")

    # what's missing is said before any engine runs
    pic = work / "cube.png"
    png(pic)
    empty = work / "empty-models"
    empty.mkdir()
    r, _ = run(["3d", "-i", str(pic), "-o", "x.glb", "--models", str(work / "nowhere")])
    if has3d:
        check(r.returncode == 2 and "no model folder" in r.stderr, "kera 3d: no model folder", r.stderr)
        r, _ = run(["3d", "-i", str(pic), "-o", "x.glb", "--models", str(empty)])
        check(r.returncode == 2 and "dinov3.gguf" in r.stderr and "tex_flow_1024.gguf" in r.stderr,
              "kera 3d: the missing files are named", r.stderr)
        r, _ = run(["3d", "-i", str(pic), "-o", "x.glb", "--models", str(empty), "--resolution", "512", "--no-texture"])
        check(r.returncode == 2 and "dinov3.gguf" in r.stderr and "tex_" not in r.stderr and
              "shape_flow_1024" not in r.stderr, "kera 3d: only the files a run needs are asked for", r.stderr)
    else:
        check(r.returncode == 2 and "built without it" in r.stderr, "kera 3d: no 3D engine is explained", r.stderr)
    r, _ = run(["3d", "-p", "a red chair", "-o", "chair.glb"])
    check(r.returncode == 2 and "--image-model" in r.stderr, "kera 3d -p without an image model says what's needed",
          r.stderr)
    r, _ = run(["3d", "-i", str(pic), "-p", "a red chair"])
    check(r.returncode == 1, "kera 3d: a picture or a description, not both", r.stderr)
    r, _ = run(["3d", "-i", str(pic), "--resolution", "768"])
    check(r.returncode == 1 and "512 or 1024" in r.stderr, "kera 3d: a wrong resolution", r.stderr)
    r, _ = run(["3d", "-i", str(pic), "--seed", "-3"])
    check(r.returncode == 1 and "--seed" in r.stderr, "kera 3d: a wrong seed", r.stderr)

    r, _ = run(["sfx", "-p", "a door creaking", "--models", str(empty)])
    if hassfx:
        check(r.returncode == 2 and "moss-soundeffect-v2-q8_0.gguf" in r.stderr, "kera sfx: no model in the folder",
              r.stderr)
        r, _ = run(["sfx", "-p", "a door creaking", "--models", str(work / "nowhere")])
        check(r.returncode == 2 and "no model folder" in r.stderr, "kera sfx: no model folder", r.stderr)
    else:
        check(r.returncode == 2 and "built without it" in r.stderr, "kera sfx: no sound-effect engine is explained",
              r.stderr)
    r, _ = run(["sfx", "-p", "a door creaking", "--duration", "-2"])
    check(r.returncode == 1 and "--duration" in r.stderr, "kera sfx: a wrong duration", r.stderr)
    if hassfx:
        # MOSS-SoundEffect's own range, said before its model is loaded (a stand-in file named as the model)
        fake = work / "fake" / "moss-soundeffect-v2-q8_0.gguf"
        fake.parent.mkdir(exist_ok=True)
        fake.write_bytes(b"GGUF")
        r, _ = run(["sfx", "-p", "a door creaking", "--duration", "45", "--model", str(fake)])
        check(r.returncode == 1 and "0.1 to 30" in r.stderr, "kera sfx: a length MOSS-SoundEffect can't make", r.stderr)
    r, _ = run(["sfx", "--duration", "3"])
    check(r.returncode == 1 and "-p" in r.stderr, "kera sfx: no description", r.stderr)

    # a real sound
    if a.sfx_model:
        def sound(seed, name):
            args = ["sfx", "-p", "a wooden door creaking open slowly", "--duration", "3", "--seed", str(seed),
                    "--model", str(a.sfx_model), "-o", name]
            if a.sfx_steps:
                args += ["--steps", str(a.sfx_steps)]
            r, t = run(args, timeout=7200)
            print(f"     kera sfx (seed {seed}) took {t:.0f} s")
            return r, work / name

        r, out = sound(5, "door.wav")
        ok = r.returncode == 0 and out.is_file()
        check(ok, "kera sfx writes a WAV", r.stderr[-3000:])
        if ok:
            rate, ch, secs, rms = wav_info(out)
            print(f"     {rate} Hz, {ch} channel(s), {secs:.2f} s, loudness {rms:.0f}")
            check(abs(secs - 3) < 0.25, "the sound is as long as asked", f"{secs:.2f} s")
            check(rms > 50, "the sound has sound in it", f"RMS {rms:.0f}")
            r2, out2 = sound(5, "door-again.wav")
            same = r2.returncode == 0 and out2.is_file() and \
                hashlib.sha256(out.read_bytes()).digest() == hashlib.sha256(out2.read_bytes()).digest()
            check(same, "the same seed makes the same file")
            r3, out3 = sound(6, "door-other.wav")
            check(r3.returncode == 0 and out3.is_file() and out3.read_bytes() != out.read_bytes(),
                  "another seed makes another sound")

    # a real 3D model
    if a.models3d and a.picture:
        def model(name):
            r, t = run(["3d", "-i", str(a.picture), "-o", name, "--models", str(a.models3d), "--resolution", "512",
                        "--seed", "7"], timeout=14400)
            print(f"     kera 3d took {t:.0f} s")
            return r, work / name

        r, out = model("thing.glb")
        ok = r.returncode == 0 and out.is_file() and out.read_bytes()[:4] == b"glTF"
        check(ok, "kera 3d writes a .glb", r.stderr[-3000:])
        if ok:
            r2, out2 = model("thing-again.glb")
            check(r2.returncode == 0 and out2.is_file() and out2.read_bytes() == out.read_bytes(),
                  "the same seed makes the same model")

    if failures:
        print("\nFAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
