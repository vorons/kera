"""Takes the build machine's folders out of the library search paths of packed programs (D-041).

  python3 packaging/strip_build_paths.py PACKAGE_DIR FOLDER [FOLDER...]

CMake adds the build folder to each program's and library's search path (RUNPATH) so that it runs from the build
tree. In a download that entry is useless, and it names the machine the download was built on. For every file in
PACKAGE_DIR, each colon-separated search path that has an entry under one of the FOLDERs is rewritten in place
without those entries and filled up with zero bytes: the file keeps its size and everything else in it its place.
Prints how many files it changed.
"""
import pathlib
import sys


def strip(data: bytearray, folders: list[bytes]) -> bool:
    changed = False
    for folder in folders:
        at = data.find(folder)
        while at >= 0:
            start = data.rfind(b"\0", 0, at) + 1
            end = data.find(b"\0", at)
            entries = bytes(data[start:end]).split(b":")
            kept = [e for e in entries if not any(e.startswith(f) for f in folders)]
            if len(kept) == len(entries):  # the folder is inside an entry, not at its start: not a search path
                at = data.find(folder, end)
                continue
            new = b":".join(kept)
            data[start:end] = new + b"\0" * (end - start - len(new))
            changed = True
            at = data.find(folder, start)
    return changed


def main():
    package = pathlib.Path(sys.argv[1])
    folders = [f.encode() for f in sys.argv[2:]]
    n = 0
    for p in sorted(package.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        data = bytearray(p.read_bytes())
        if data[:4] == b"\x7fELF" and strip(data, folders):
            p.write_bytes(data)
            n += 1
    print(f"search paths cleaned in {n} file(s)")


if __name__ == "__main__":
    main()
