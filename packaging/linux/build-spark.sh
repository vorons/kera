#!/bin/bash
# Builds, checks and packs the DGX Spark download on a DGX Spark itself (D-041), by the recipe of
# .github/workflows/build-linux-arm64-cuda.yml.
#
#   [KERA_COMMIT=abc123] packaging/linux/build-spark.sh VERSION [BUILD_DIR]      e.g. v0.1.5
#
# KERA_COMMIT names the commit in `kera version` when the source is a snapshot without its git history.
#
# Needs no admin rights: cmake, a C++ compiler and NVIDIA's CUDA 13 toolkit, as DGX OS has them. The folder names of
# the machine it is built on are kept out of the programs (-ffile-prefix-map), and the package is searched for them
# before it is packed. Writes kera-VERSION-linux-arm64-cuda.tar.gz into BUILD_DIR.
set -euo pipefail
v=${1:?the version, e.g. v0.1.5}
src=$(cd "$(dirname "$0")/../.." && pwd)
build=${2:-$src/build-spark}
export PATH=/usr/local/cuda/bin:$PATH
map="-ffile-prefix-map=$src=/kera -ffile-prefix-map=$build=/kera/build"
export CFLAGS="$map" CXXFLAGS="$map" CUDAFLAGS="-Xcompiler=-ffile-prefix-map=$src=/kera -Xcompiler=-ffile-prefix-map=$build=/kera/build"
compilers=()
if command -v gcc-14 > /dev/null; then
    # GCC 14: the CPU variant for the newest ARM chips needs +sme, which GCC 13 doesn't know
    compilers=(-DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DCMAKE_CUDA_HOST_COMPILER=g++-14)
fi
generator=()
if command -v ninja > /dev/null; then
    generator=(-G Ninja)
fi
cmake -S "$src" -B "$build" "${generator[@]}" -DCMAKE_BUILD_TYPE=Release "${compilers[@]}" \
    -DBUILD_SHARED_LIBS=ON -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real -DGGML_CUDA_CCCL_VERSION=v3.4.3 \
    -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined \
    -DLLAMA_OPENSSL=OFF -DKERA_WARNINGS_AS_ERRORS=ON -DKERA_SOURCE_DIR_IN_PROGRAM=/kera \
    -DKERA_COMMIT="${KERA_COMMIT:-}"
# without Ninja the engines' own builds (image, music, 3D, sound effects) would compile one file at a time: this
# gives each of them several cores too
export CMAKE_BUILD_PARALLEL_LEVEL="${KERA_ENGINE_JOBS:-$(( ($(nproc) + 1) / 2 ))}"
cmake --build "$build" -j "${KERA_BUILD_JOBS:-$(nproc)}"
ctest --test-dir "$build" --output-on-failure

bin=$build/bin
for f in libggml-cuda.so image/libggml-cuda.so music/libggml-cuda.so 3d/libggml-cuda.so 3d/trellis-cli sfx/libggml-cuda.so; do
    readelf -d "$bin/$f" | grep -q /usr/local/cuda/lib64 || { echo "$f: NVIDIA's folder is not in its library path"; exit 1; }
done
"$bin/kera" version

name=kera-$v-linux-arm64-cuda
out=$build/$name
rm -rf "${out:?}"
mkdir -p "$out/image" "$out/music" "$out/3d" "$out/sfx"
cp "$bin/kera" "$bin"/*.so* "$out/"
cp "$bin/image/sd-cli" "$bin/image/sd-server" "$out/image/"
cp -P "$bin/image/libstable-diffusion.so" "$bin"/image/libggml* "$out/image/"
cp "$bin/music/ace-lm" "$bin/music/ace-synth" "$bin/music/ace-server" "$out/music/"
cp -P "$bin"/music/libggml* "$out/music/"
cp "$bin/3d/trellis-cli" "$out/3d/"
cp -P "$bin"/3d/libggml* "$out/3d/"
cp "$bin/sfx/audiocpp_cli" "$bin/sfx/audiocpp_server" "$out/sfx/"
cp -P "$bin"/sfx/libggml* "$out/sfx/"
cp "$src/packaging/linux/README-spark.txt" "$out/README.txt"
cp "$src/LICENSE" "$src/NOTICE" "$build/generated/THIRD_PARTY_LICENSES.txt" "$out/"
python3 "$src/packaging/copy_finetune.py" "$out"
# the programs as packed, from the packed folder
says() { # the word a program's --help must print (some end with an error code after printing it)
    local word=$1
    shift
    local text
    text=$("$@" --help 2>&1 || true)
    grep -q -- "$word" <<< "$text" || { echo "$1 --help does not say $word"; exit 1; }
}
"$out/image/sd-cli" --help > /dev/null
says "Usage:" "$out/music/ace-synth"
says "usage:" "$out/3d/trellis-cli"
says "--task" "$out/sfx/audiocpp_cli"
# the build folder goes out of the programs' library search paths (CMake adds it so that they run from the build
# tree); they are run once more after, as packed
python3 "$src/packaging/strip_build_paths.py" "$out" "$build" "$src"
"$out/kera" version
"$out/image/sd-cli" --help > /dev/null
says "Usage:" "$out/music/ace-synth"
says "usage:" "$out/3d/trellis-cli"
says "--task" "$out/sfx/audiocpp_cli"
# nothing of this machine in the package: its folders, its home folder (which holds its login)
for word in "$src" "$build" "$HOME"; do
    if grep -r -l -a -F -- "$word" "$out" > /dev/null; then
        echo "the package names this machine ($(grep -r -l -a -F -- "$word" "$out" | head -5 | xargs -n1 basename | tr '\n' ' ')): not packed"
        exit 1
    fi
done
tar -C "$build" -czf "$build/$name.tar.gz" "$name"
ls -la "$build/$name.tar.gz"
