"""Copies what `kera finetune` needs into a download (spec Phase 10, M11): Kera's fine-tuning script and llama.cpp's
converters (LoRA and whole models), into DEST/finetune.

  python packaging/copy_finetune.py DEST
"""
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LLAMA = ROOT / "third_party" / "llama.cpp"


def main():
    dest = pathlib.Path(sys.argv[1]) / "finetune"
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "llama.cpp").mkdir(parents=True)
    for f in ("kera_finetune.py", "requirements.txt"):
        shutil.copy2(ROOT / "tools" / "finetune" / f, dest / f)
    for f in ("convert_lora_to_gguf.py", "convert_hf_to_gguf.py", "LICENSE"):
        shutil.copy2(LLAMA / f, dest / "llama.cpp" / f)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "tests")
    shutil.copytree(LLAMA / "conversion", dest / "llama.cpp" / "conversion", ignore=ignore)
    shutil.copytree(LLAMA / "gguf-py" / "gguf", dest / "llama.cpp" / "gguf-py" / "gguf", ignore=ignore)
    n = sum(1 for _ in dest.rglob("*") if _.is_file())
    print(f"copied {n} files to {dest}")


if __name__ == "__main__":
    main()
