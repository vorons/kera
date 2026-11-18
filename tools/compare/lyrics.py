"""How many of a song's given words can be heard: Whisper's transcript of each song against its lyrics.

  pip install faster-whisper numpy
  python tools/compare/lyrics.py --songs DIR --lyrics DIR

DIR/<kind>-<lyrics number>-<seed>.wav are songs made with `kera music --lyrics @lyrics/l<number>.txt` (48 kHz WAV);
<kind> names what is compared ("new", "old"...). Prints, per song and per kind, the lyrics' words found in order in
the transcript (difflib's matching blocks), and the same counted against the longer of the two, so words the song
adds count against it. Used for docs/QUALITY.md's music rows (D-036). Whisper mishears sung words too, so the numbers
compare settings on the same lyrics and seeds; they aren't a song's true error rate.
"""
import argparse
import difflib
import glob
import os
import re
import wave

import numpy as np
from faster_whisper import WhisperModel


def words(s):
    return re.findall(r"[a-z']+", re.sub(r"\[[^\]]*\]", " ", s.lower()))


def load16k(path):
    """A 48 kHz 16-bit WAV as 16 kHz mono floats: low-passed at 7 kHz, then every third sample."""
    w = wave.open(path)
    sr, ch = w.getframerate(), w.getnchannels()
    if sr != 48000 or w.getsampwidth() != 2:
        raise SystemExit(f"{path}: a 48 kHz 16-bit WAV is expected")
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32).reshape(-1, ch).mean(1) / 32768.0
    t = np.arange(-60, 61)
    fir = np.sinc(2 * 7000 / sr * t) * (2 * 7000 / sr) * np.hamming(len(t))
    return np.convolve(a, fir, mode="same")[::3].astype(np.float32), len(a) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", required=True)
    ap.add_argument("--lyrics", required=True)
    ap.add_argument("--model", default="medium.en")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    m = WhisperModel(a.model, device="cpu", compute_type="int8", cpu_threads=a.threads)
    total = {}
    for f in sorted(glob.glob(os.path.join(a.songs, "*.wav"))):
        name = os.path.basename(f)[:-4]
        kind, number, _seed = name.rsplit("-", 2)
        ref = words(open(os.path.join(a.lyrics, f"l{number}.txt"), encoding="utf-8").read())
        audio, secs = load16k(f)
        segments, _ = m.transcribe(audio, language="en", beam_size=5)
        heard = words(" ".join(s.text for s in segments))
        hit = sum(b.size for b in difflib.SequenceMatcher(None, ref, heard).get_matching_blocks())
        t = total.setdefault(kind, [0, 0, 0])
        t[0] += hit
        t[1] += len(ref)
        t[2] += max(len(ref), len(heard))
        print(f"{name:16s} words right {hit}/{len(ref)}  heard {len(heard)}  length {secs:.0f} s", flush=True)
    for kind, (hit, ref, longer) in sorted(total.items()):
        print(f"TOTAL {kind:8s} words right {hit}/{ref} ({100 * hit / ref:.0f}%), with added words counted {100 * hit / longer:.0f}%")


if __name__ == "__main__":
    main()
