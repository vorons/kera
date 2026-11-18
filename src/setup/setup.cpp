// Auto-setup steps 1-2 (spec 4a): detect the machine, then measure it, with progress for the app.

#include "setup/kr_machine.h"

#include "common/kr_paths.h"
#include "tune/kr_settings.h"

#include "kera_version.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <cctype>
#include <thread>

#if !defined(_WIN32)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kera {

// An NVIDIA chip sharing the computer's memory (the DGX Spark's GB10)
static bool shared_cuda(const machine_profile & p) {
    return !p.gpus.empty() && p.gpus[0].backend == "CUDA" && p.gpus[0].unified;
}

static double estimate_usable_gb(const machine_profile & p) {
    // A model shares memory with the OS and other apps. These are starting estimates; the tuner refines them.
    double usable = std::max(0.0, 0.8 * p.ram_gb - 1.0); // 64 GB -> ~50 GB (spec section 2)
    if (p.os_family == "linux" && shared_cuda(p) && p.ram_gb >= 64) {
        // a DGX Spark runs little beside the model: about 110 GB of its 119.5 GiB (spec section 2, headless)
        usable = std::max(usable, p.ram_gb - 10.0);
    }
    if (p.os_family == "macos") {
        // macOS caps how much memory the graphics chip may wire (verify iogpu.wired_limit_mb); 16 GB -> ~10 GB
        double cap = 0.65 * p.ram_gb;
        if (!p.gpus.empty() && p.gpus[0].total_gb > 0) {
            cap = p.gpus[0].total_gb;
        }
        usable = std::min(usable, cap);
    }
    return usable;
}

// Is the folder `path` on the drive mounted at `mount`? By device, not by name: on macOS the Kera
// folder (/Users/...) lives on /System/Volumes/Data through a firmlink, so no path prefix matches.
static bool on_drive(const std::string & path, const std::string & mount) {
#if defined(_WIN32)
    return path.size() >= 2 && mount.size() >= 2 && path[1] == ':' && mount[1] == ':' &&
           std::toupper((unsigned char) path[0]) == std::toupper((unsigned char) mount[0]);
#else
    struct stat a {}, b {};
    if (stat(path.c_str(), &a) == 0 && stat(mount.c_str(), &b) == 0) {
        return a.st_dev == b.st_dev;
    }
    return path.rfind(mount, 0) == 0;
#endif
}

// The folder for a drive's speed-test file: the Kera folder on the drive that holds it (always
// writable), otherwise a folder at the top of the drive.
static std::string test_dir_for(const drive_profile & d, const std::string & home) {
    return on_drive(home, d.path) ? join(home, "tmp") : join(d.path, ".kera-speed-test");
}

// The DGX Spark's setup advice (spec 10.6), in plain words. Kera never changes system settings itself: each needs
// the owner's admin rights, so it only says what would help.
static void spark_notes(machine_profile & p) {
#if defined(__linux__)
    // NVIDIA's driver is here but CUDA didn't start: the CUDA libraries Kera's CUDA part needs (cuBLAS 13) are missing
    std::error_code dec;
    const bool nvidia_driver = std::filesystem::exists("/proc/driver/nvidia/version", dec);
    const bool any_cuda = std::any_of(p.gpus.begin(), p.gpus.end(), [](const gpu_profile & g) { return g.backend == "CUDA"; });
    if (nvidia_driver && !any_cuda && std::filesystem::exists(join(exe_dir(), "libggml-cuda.so"), dec)) {
        p.notes.push_back("NVIDIA's driver is installed, but Kera couldn't start CUDA, so models run on the CPU. "
                          "CUDA 13's libraries (cudart and cuBLAS) are probably missing or not on the library path: "
                          "on DGX OS they're in /usr/local/cuda/lib64.");
    }
    if (!shared_cuda(p) || p.ram_gb < 64) {
        return;
    }
    // DGX OS 7.5.0's kernel feature KHO leaves about 112 GiB of the 128 GB; kho=off or NVIDIA's fix gives 119.5
    if (p.ram_gb > 100 && p.ram_gb < 116) {
        p.notes.push_back("Linux sees " + std::to_string((int) (p.ram_gb + 0.5)) + " GB of this DGX Spark's 128 GB. DGX OS 7.5.0 keeps about 7.5 GB "
                          "back for a kernel feature (KHO): NVIDIA's update, or the boot option kho=off, gives it "
                          "back. Kera doesn't change boot settings.");
    }
    if (std::ifstream f("/proc/sys/vm/swappiness"); f) {
        int swappiness = -1;
        f >> swappiness;
        if (swappiness > 10) {
            p.notes.push_back("Linux may move parts of a loaded model to swap (vm.swappiness is " +
                              std::to_string(swappiness) + "). For big models, 1 keeps them in memory: "
                              "sudo sysctl vm.swappiness=1. Kera doesn't change it.");
        }
    }
    // a desktop session takes about 2-3 GB a model could use
    std::error_code ec;
    const std::string wayland = "/run/user/" + std::to_string(getuid()) + "/wayland-0";
    if (std::filesystem::exists("/tmp/.X11-unix/X0", ec) || std::filesystem::exists(wayland, ec)) {
        p.notes.push_back("The desktop is running. Without it (headless, reached over the network) a model gets about "
                          "2-3 GB more.");
    }
#else
    (void) p;
#endif
}

machine_profile run_setup(const setup_options & opt) {
    kr_settings_register_builtin();

    machine_profile p;
    p.kera_version = KERA_VERSION;
    p.measured_at = utc_now_iso8601();

    auto progress = [&](double f, const std::string & step) {
        if (opt.progress) opt.progress(f, step);
    };

    progress(0.02, "Looking at this computer");
    detect_system(p);
    p.drives = list_drives();

    // Which drives to test: the one holding the Kera folder first, then the others.
    const std::string home = home_dir();
    std::stable_partition(p.drives.begin(), p.drives.end(), [&](const drive_profile & d) { return on_drive(home, d.path); });

    progress(0.05, "Measuring memory speed");
    const int threads = p.cpu_cores > 0 ? p.cpu_cores : (int) std::max(1u, std::thread::hardware_concurrency());
    const size_t ram_bytes = opt.quick ? (64u << 20) : (size_t) kr_setting_get("setup.ram_test_mib") * (1u << 20);
    p.ram_gbps.cpu = measure_cpu_read_gbps(ram_bytes, threads, opt.quick ? 0.2 : 1.5);

    progress(0.15, "Measuring the graphics chip");
    {
        setup_options o = opt;
        o.progress = [&](double, const std::string & step) { progress(0.2, step); };
        measure_gpus(p, o);
    }

    p.usable_gb = estimate_usable_gb(p);
    spark_notes(p);

    if (opt.measure_disks) {
        std::vector<drive_profile *> to_test;
        for (auto & d : p.drives) {
            if (!opt.only_drives.empty() &&
                std::find(opt.only_drives.begin(), opt.only_drives.end(), d.path) == opt.only_drives.end()) {
                continue;
            }
            to_test.push_back(&d);
        }
        const size_t max_drives = opt.quick ? 1 : 4;
        for (size_t i = 0; i < to_test.size(); ++i) {
            drive_profile & d = *to_test[i];
            if (i >= max_drives) {
                d.note = opt.quick ? "The quick setup tests one drive; the full setup tests up to four."
                                   : "Kera tests up to " + std::to_string(max_drives) + " drives.";
                continue;
            }
            progress(0.35 + 0.6 * i / to_test.size(), "Measuring the SSD speed of " + d.path);
            measure_drive(d, test_dir_for(d, home), opt);
        }
    }

    progress(1.0, "Done");
    return p;
}

} // namespace kera
