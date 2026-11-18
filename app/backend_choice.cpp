// `--backend auto|vulkan|hip|cpu` for kera serve and kera exactness (spec Phase 8, M1 risk test 5).
//
// The Windows download runs on Vulkan. AMD's ROCm (HIP) back-end is an optional add-on: its files sit in a `hip`
// folder next to Kera, which llama.cpp doesn't scan, so nothing changes unless it's asked for. `--backend hip`
// loads it and uses only its device; the folder is also put on Windows' DLL search path, so AMD's runtime there is
// found before the older copy AMD's graphics driver keeps in System32 (as llama.cpp's own release does by placing it
// next to the program). Vulkan stays the default until HIP passes the exactness tests and wins on the Minisforum.

#include "app/kera_app.h"
#include "common/kr_paths.h"

#include "ggml-backend.h"

#include <cstdio>
#include <filesystem>
#include <string>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace kera {

namespace {

fs::path hip_library() {
#if defined(_WIN32)
    return fs::u8path(exe_dir()) / "hip" / "ggml-hip.dll";
#else
    return fs::u8path(exe_dir()) / "hip" / "libggml-hip.so";
#endif
}

} // namespace

bool hip_addon_installed() {
    std::error_code ec;
    return fs::is_regular_file(hip_library(), ec);
}

bool choose_backend(const std::string & which, std::string & device, std::string * error) {
    device.clear();
    if (which.empty() || which == "auto") {
        return true; // llama.cpp's choice: every graphics chip it finds
    }
    if (which == "cpu") {
        device = "none";
        return true;
    }
    if (which == "vulkan") {
        device = "Vulkan0";
        return true;
    }
    if (which != "hip") {
        if (error) *error = "--backend takes auto, auto-model, vulkan, hip or cpu";
        return false;
    }
    const fs::path lib = hip_library();
    const fs::path dir = lib.parent_path();
    std::error_code ec;
    if (!fs::is_regular_file(lib, ec)) {
        if (error) *error = "the ROCm (HIP) add-on isn't installed: unzip it into Kera's folder so that " + lib.u8string() + " exists";
        return false;
    }
#if defined(_WIN32)
    // rocBLAS stops the whole program at its first sum when its kernels are missing (the first add-on with AMD's
    // maths libraries lacked them, N-007): its folder in the add-on, and the .kpack folder next to "hip"
    if (!fs::is_directory(dir / "rocblas" / "library", ec) || !fs::is_directory(dir.parent_path() / ".kpack", ec)) {
        if (error) {
            *error = "the ROCm (HIP) add-on is missing AMD's kernel files (hip\\rocblas\\library and .kpack): download it "
                     "again and unzip all of it into Kera's folder";
        }
        return false;
    }
    SetDllDirectoryW(dir.wstring().c_str()); // AMD's runtime from the add-on, not the driver's copy in System32
    // Windows says why a DLL doesn't load; ggml doesn't pass it on. Error 126 means a file the add-on needs is missing
    // (preview 2's lacked hipBLAS), which is the download's fault, not the driver's.
    if (LoadLibraryW(lib.wstring().c_str()) == nullptr) {
        const DWORD e = GetLastError();
        if (error) {
            *error = e == ERROR_MOD_NOT_FOUND
                ? "the ROCm (HIP) add-on is missing a file it needs (Windows error 126): download it again"
                : "the ROCm (HIP) back-end didn't load (Windows error " + std::to_string(e) +
                  "): is the AMD graphics driver up to date?";
        }
        return false;
    }
#endif
    if (ggml_backend_load(lib.u8string().c_str()) == nullptr) {
        if (error) *error = "the ROCm (HIP) back-end didn't load: is the AMD graphics driver up to date?";
        return false;
    }
    device = "ROCm0";
    return true;
}

} // namespace kera
