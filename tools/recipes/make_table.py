"""Writes app/image_recipes.inc: the picture and video models `kera image` and `kera video` recognise by file name
beyond the hand-written ones in app/image_cli.cpp (D-036).

  python tools/recipes/make_table.py            # rewrite app/image_recipes.inc
  python tools/recipes/make_table.py --check    # fail if it is out of date (CI)

Each row is one of the image engine's own recipes (third_party/stable-diffusion.cpp/examples/server/recipes/<id>.json:
CFG, guidance, size, flow shift, sampler and schedule, each reviewed there against the makers' cards) with the makers'
own step count from tools/recipes/makers.json in place of the recipes' cap of 20.
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RECIPES = ROOT / "third_party" / "stable-diffusion.cpp" / "examples" / "server" / "recipes"
OUT = ROOT / "app" / "image_recipes.inc"


def key(name):
    return "".join(c for c in name.lower() if c not in "-_. ")


def num(v):
    return "-1" if v is None else repr(float(v)) if isinstance(v, float) and v != int(v) else str(int(v))


def cstr(s):
    return "nullptr" if s is None else json.dumps(s, ensure_ascii=True)


def table():
    makers = json.loads((ROOT / "tools" / "recipes" / "makers.json").read_text(encoding="utf-8"))
    rows, seen = [], {}
    for m in makers["models"]:
        d = dict(json.loads((RECIPES / (m["recipe"] + ".json")).read_text(encoding="utf-8")).get("defaults", {}))
        for k in m.get("drop", []):
            d.pop(k, None)
        steps = m.get("steps", d.get("steps"))
        if steps is None:
            sys.exit(f"{m['recipe']}: no step count")
        names = sorted({key(n) for n in m["names"]})
        for n in names:
            if n in seen:
                sys.exit(f"{n}: named by {seen[n]} and {m['recipe']}")
            seen[n] = m["recipe"]
        video = bool(m.get("video"))
        shift = m.get("shift", d.get("flow_shift"))
        rows.append("    { %s, %s, %d, %s, %s, %s, %d, %d, %s, %s, %s, %s }," % (
            cstr(m["family"]), cstr("|" + "|".join(names) + "|"), steps, num(d.get("cfg_scale")),
            num(d.get("distilled_guidance")), num(shift), 0 if video else d.get("width") or 0,
            0 if video else d.get("height") or 0, cstr(d.get("sampling_method")), cstr(d.get("scheduler")),
            "true" if m.get("with_picture") else "false", "true" if video else "false"))
    head = ("// Made by tools/recipes/make_table.py from tools/recipes/makers.json and the image engine's recipes: do not edit.\n"
            "// family, |names|, steps, CFG, guidance, flow shift, width, height, sampler, schedule, steps with a picture, video\n")
    return head + "\n".join(rows) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    text = table()
    if a.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            sys.exit("app/image_recipes.inc is out of date: run python tools/recipes/make_table.py")
        print("app/image_recipes.inc is up to date")
        return
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({text.count(chr(10)) - 2} models)")


if __name__ == "__main__":
    main()
