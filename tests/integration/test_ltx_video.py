"""LTX makes a real video: frames with a picture in them, at the model's frame rate, with sound (the owner's report of
10 Nov 2026: LTX 2.5 stopped at the first step, then gave white frames, and nothing here made an actual LTX video).

  python tests/integration/test_ltx_video.py --kera build/bin/kera --dit FILE --llm FILE --vae FILE [--audio-vae FILE]
                                             [--frames 33] [--steps 4] [--type q8_0] [--keep OUT.avi] [--follows]

Needs the model's files (tens of GB: not downloaded here, so this runs where they are, on a graphics chip) and ffmpeg
and ffprobe. The command line sets no engine switches: the video is made with Kera's own choices. Checks:
- kera video ends without an error;
- the file has the frames asked for at 24 frames a second;
- frames near the start, the middle and the end are pictures: not one flat colour, not white, not black;
- with --audio-vae, there is a sound track that is neither silent nor at full scale throughout;
- the words reach the video (the owner's bug 3, 11 Nov 2026: a real video, but of something else): with --follows,
  two more clips with the same seed, one asked to be bright (a snowfield at noon) and one dark (a cave at night).
  They must differ, and the bright one must be clearly brighter. A video made without its words fails this
  whatever it shows.

The fault of 11 Nov needed the long clip at the model's own precision: run it as the owner does, with --frames 121
--steps 20 and no --type, where the machine has the memory (about 95 GB in use at the peak).
"""
import argparse
import pathlib
import re
import subprocess
import sys
import tempfile

failures = []


def check(ok, what, detail=""):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failures.append(what)
        if detail:
            print("     " + detail.replace("\n", "\n     ")[-2500:])


def out_of(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kera", required=True)
    ap.add_argument("--dit", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--audio-vae")
    ap.add_argument("--frames", type=int, default=33)
    ap.add_argument("--steps", type=int, default=4, help="few, to be quick: enough to tell a picture from a blank")
    ap.add_argument("--keep", type=pathlib.Path, help="keep the video here")
    ap.add_argument("--type", help="load the weights at this precision (e.g. q8_0), where memory is short")
    ap.add_argument("--follows", action="store_true", help="also check that the words reach the video (two more clips)")
    a = ap.parse_args()

    work = pathlib.Path(tempfile.mkdtemp(prefix="kera-ltx-"))
    video = a.keep or work / "ltx.avi"

    def make(prompt, out):
        cmd = [a.kera, "video", "--diffusion-model", a.dit, "--llm", a.llm, "--vae", a.vae, "-p", prompt,
               "-W", "768", "-H", "512", "--video-frames", str(a.frames), "--steps", str(a.steps), "--seed", "42",
               "-o", str(out)]
        if a.audio_vae:
            cmd += ["--audio-vae", a.audio_vae]
        if a.type:
            cmd += ["--type", a.type]
        return out_of(cmd)

    def brightness(path, n):
        stats = out_of(["ffmpeg", "-v", "error", "-i", str(path), "-vf",
                        f"select=eq(n\\,{n}),signalstats,metadata=print:file=-", "-f", "null", "-"]).stdout
        m = re.search(r"signalstats\.YAVG=([0-9.]+)", stats)
        return float(m.group(1)) if m else None

    r = make("A sea turtle glides slowly over a bright coral reef in clear blue water, sunbeams flickering through "
             "the surface. Gentle underwater ambience with soft bubbles.", video)
    log = r.stdout + r.stderr
    check(r.returncode == 0 and video.is_file(), "kera video made a file", log)
    if not video.is_file():
        sys.exit(1)
    check("24 frames a second" in log, "Kera chose 24 frames a second by itself", log[:600])

    streams = out_of(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,r_frame_rate,nb_frames", "-of",
                      "csv=p=0", str(video)]).stdout.split()
    v = [s.split(",") for s in streams if s.startswith("video")]
    check(len(v) == 1 and v[0][1] == "24/1" and v[0][2] == str(a.frames),
          f"the file has {a.frames} frames at 24 a second", str(streams))

    for n in sorted({2, a.frames // 2, a.frames - 3}):
        stats = out_of(["ffmpeg", "-v", "error", "-i", str(video), "-vf",
                        f"select=eq(n\\,{n}),signalstats,metadata=print:file=-", "-f", "null", "-"]).stdout
        g = {k: float(x) for k, x in re.findall(r"signalstats\.(YMIN|YAVG|YMAX)=([0-9.]+)", stats)}
        ok = len(g) == 3 and g["YMAX"] - g["YMIN"] >= 40 and 20 <= g["YAVG"] <= 235
        check(ok, f"frame {n} is a picture (brightness {g.get('YMIN')} to {g.get('YMAX')}, mean {g.get('YAVG')})", stats)

    if a.audio_vae:
        check(any(s.startswith("audio") for s in streams), "the file has a sound track", str(streams))
        vol = out_of(["ffmpeg", "-v", "info", "-i", str(video), "-vn", "-af", "volumedetect", "-f", "null", "-"]).stderr
        m = re.search(r"mean_volume: (-?[0-9.]+|-inf) dB", vol)
        mean = float(m.group(1)) if m and m.group(1) != "-inf" else float("-inf")
        check(-70 < mean < -3, f"the sound is neither silent nor at full scale (mean {mean} dB)", vol[-600:])

    if a.follows:
        clips = {}
        for name, prompt in (("bright", "A wide snowfield at noon under a white overcast sky, everything covered in "
                                        "bright white snow, a few snowflakes drifting. Quiet wind."),
                             ("dark", "The inside of a dark cave at night, black rock walls, lit only by one small "
                                      "candle flame in the distance. Dripping water.")):
            out = (a.keep.with_name(a.keep.stem + "-" + name + ".avi") if a.keep else work / f"{name}.avi")
            r = make(prompt, out)
            check(r.returncode == 0 and out.is_file(), f"kera video made the {name} clip", r.stdout + r.stderr)
            if out.is_file():
                means = [brightness(out, n) for n in sorted({2, a.frames // 2, a.frames - 3})]
                clips[name] = sum(means) / len(means) if all(m is not None for m in means) else None
        b, d = clips.get("bright"), clips.get("dark")
        check(b is not None and d is not None and b - d >= 40,
              f"the words reach the video: the snowfield is brighter than the cave (mean {b} against {d})")

    print("\n" + ("FAIL: " + "; ".join(failures) if failures else "PASS: LTX made a real video at Kera's own settings"))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
