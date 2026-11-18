"""Publishes Kera on Hugging Face (the owner's decision of 4 Nov 2026, docs/DECISIONS.md D-026): a public model repo
under Kera Labs with the downloads of one release, the source code of the commit it was built from (a snapshot,
without the git history), the page (tools/publish/huggingface_README.md), LICENSE, NOTICE and two charts.

  HF_TOKEN=... python tools/publish/huggingface.py --version v0.1.0-preview.4 --downloads DIR [--repo KeraAI/Kera]
                                                   [--private] [--dry-run]

DIR holds the release's files (as GitHub's release has them: kera-VERSION-windows-x64.zip, its -hip-addon.zip,
kera-VERSION-linux-x64.tar.gz) and Kera-vs-Ollama.dmg. The token needs write access to the repo's owner (a Hugging
Face organisation for Kera Labs, made once on huggingface.co). --dry-run builds the folder and lists it without
uploading. Needs `pip install huggingface_hub`.
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CHARTS = {
    "long-documents.png": "bench/results/minisforum/2026-10-16-nemotron-3-nano-long-all-engines.png",
}


def fill_page(v, spark_v):
    """The page, with the version of the downloads and, where it is newer, the DGX Spark download's."""
    page = (ROOT / "tools/publish/huggingface_README.md").read_text(encoding="utf-8")
    note = source = ""
    if spark_v != v:
        n = spark_v.lstrip("v")
        note = (f"**The DGX Spark download is {n}, the newest** (built and tested on a DGX Spark): LTX video "
                f"works and follows what you ask, and music loads about 15 times sooner there, with the same songs. The other downloads are {v.lstrip('v')} until they "
                f"are built; `source/kera-{spark_v}-source.tar.gz` is the Spark download's source.\n\n")
        source = f"\n| `source/kera-{spark_v}-source.tar.gz` | The source code of the DGX Spark download |"
    return (page.replace("@SPARK_VERSION@", spark_v).replace("@SPARK_SOURCE@", source).replace("@SPARK_NOTE@", note)
            .replace("@VERSION@", v))


def spark_only(a):
    """Only the DGX Spark download of a.version: the other downloads and their source stay as they are."""
    v, others = a.version, a.spark_only
    name = f"kera-{v}-linux-arm64-cuda.tar.gz"
    if not (a.downloads / name).is_file():
        sys.exit(f"missing from --downloads: {name}")
    stage = pathlib.Path(tempfile.mkdtemp(prefix="kera-hf-"))
    (stage / "downloads").mkdir()
    (stage / "source").mkdir()
    (stage / "README.md").write_text(fill_page(others, v), encoding="utf-8")
    shutil.copy(ROOT / "docs" / "MODELS.md", stage / "MODELS.md")
    shutil.copy(a.downloads / name, stage / "downloads" / name)
    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar.gz", f"--prefix=kera-{v}/", "-o",
                    str(stage / "source" / f"kera-{v}-source.tar.gz"), a.commit], check=True)
    for p in sorted(stage.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:9.1f} MB  {p.relative_to(stage)}")
    if a.dry_run:
        print(f"\n(dry run: nothing uploaded; the folder is {stage})")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN isn't set")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    # the earlier Spark download goes; the other downloads and their source stay
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(stage),
                      commit_message=f"Kera {v} for the DGX Spark", delete_patterns=["downloads/kera-*-linux-arm64-cuda.tar.gz"])
    print(f"\npublished https://huggingface.co/{a.repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="the release, e.g. v0.1.0-preview.4")
    ap.add_argument("--downloads", required=True, type=pathlib.Path, help="the folder with the release's files")
    ap.add_argument("--repo", default="KeraAI/Kera")
    ap.add_argument("--commit", default="HEAD", help="the commit the release was built from")
    ap.add_argument("--private", action="store_true", help="make the repo private (the owner chose public)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--spark-only", metavar="OTHERS", help="publish only the DGX Spark download of --version; the "
                    "other downloads on the page stay at version OTHERS (e.g. v0.1.4), which the page then names")
    a = ap.parse_args()
    if a.spark_only:
        return spark_only(a)

    v = a.version
    wanted = [f"kera-{v}-windows-x64.zip", f"kera-{v}-windows-x64-hip-addon.zip", f"kera-{v}-linux-x64.tar.gz"]
    missing = [f for f in wanted if not (a.downloads / f).is_file()]
    if missing:
        sys.exit("missing from --downloads: " + ", ".join(missing))
    # the Mac and DGX Spark downloads (full releases) and the Kera-vs-Ollama app (built on its own): uploaded when given; an app
    # already on the page stays
    wanted += [f for f in (f"kera-{v}-macos-arm64.zip", f"kera-{v}-linux-arm64-cuda.tar.gz", "Kera-vs-Ollama.dmg")
               if (a.downloads / f).is_file()]

    stage = pathlib.Path(tempfile.mkdtemp(prefix="kera-hf-"))
    (stage / "downloads").mkdir()
    (stage / "source").mkdir()
    (stage / "images").mkdir()
    page = fill_page(v, v)
    (stage / "README.md").write_text(page, encoding="utf-8")
    for f in ("LICENSE", "NOTICE"):
        shutil.copy(ROOT / f, stage / f)
    shutil.copy(ROOT / "docs" / "MODELS.md", stage / "MODELS.md")  # the supported models (tools/publish/models_list.py)
    for name, src in CHARTS.items():
        shutil.copy(ROOT / src, stage / "images" / name)
    for f in wanted:
        shutil.copy(a.downloads / f, stage / "downloads" / f)
    # the source as it is at that commit, without the history (git archive)
    src = stage / "source" / f"kera-{v}-source.tar.gz"
    subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar.gz", f"--prefix=kera-{v}/", "-o", str(src),
                    a.commit], check=True)

    for p in sorted(stage.rglob("*")):
        if p.is_file():
            print(f"{p.stat().st_size / 1e6:9.1f} MB  {p.relative_to(stage)}")
    if a.dry_run:
        print(f"\n(dry run: nothing uploaded; the folder is {stage})")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN isn't set")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(a.repo, repo_type="model", private=a.private, exist_ok=True)
    # the page lists one version: an earlier version's downloads and source go
    api.upload_folder(repo_id=a.repo, repo_type="model", folder_path=str(stage), commit_message=f"Kera {v}",
                      delete_patterns=["downloads/kera-*", "source/*", "images/*"])  # a chart no longer shown goes too
    print(f"\npublished https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
