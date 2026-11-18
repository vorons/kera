// Machine profile (spec P3, section 4a): written by auto-setup, read by the planner and the tuner.
// Saved as kera_machine.json in the Kera folder.
#pragma once

#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace kera {

struct drive_profile {
    std::string path;            // mount point or drive letter, e.g. "C:\" or "/"
    std::string device;          // e.g. "/dev/nvme0n1p2" where the OS tells us
    double total_gb   = 0.0;
    double free_gb    = 0.0;
    bool   measured   = false;
    double read_gbps  = 0.0;     // best parallel direct-read speed
    double write_gbps = 0.0;     // writing the test file
    int    best_read_mib = 0;    // read size that gave read_gbps
    int    best_inflight = 0;    // reads in flight that gave read_gbps
    double single_read_gbps = 0.0; // one read at a time, 1 MiB: what simple tests report
    bool   direct = false;       // the OS file cache was bypassed
    std::string note;            // plain words when something was skipped or looks wrong
};

struct gpu_profile {
    std::string name;            // e.g. "AMD Radeon 8060S Graphics"
    std::string backend;         // e.g. "Vulkan", "Metal", "CUDA"
    double total_gb     = 0.0;   // memory the graphics chip may use
    double free_gb      = 0.0;
    double max_alloc_gb = 0.0;
    bool   unified      = false; // shares memory with the CPU (Apple, Ryzen AI Max, GB10)
};

struct machine_profile {
    std::string kera_version;
    std::string measured_at;     // UTC, ISO 8601
    std::string os;              // e.g. "Windows 11 (build 26100)"
    std::string os_family;       // "windows", "macos", "linux"
    std::string chip;            // CPU name
    int cpu_cores   = 0;         // physical cores
    int cpu_threads = 0;
    double ram_gb    = 0.0;
    double usable_gb = 0.0;      // memory a model may use (estimate; the tuner refines it)
    std::vector<gpu_profile> gpus;

    struct {
        double gpu  = 0.0;       // graphics chip reading its memory
        double cpu  = 0.0;       // CPU reading RAM, all cores
        double both = 0.0;       // both at once
    } ram_gbps;

    double gpu_q4_gbps = 0.0;    // 4-bit matrix-times-vector on the graphics chip, weight bytes per second
    double cpu_q4_gbps = 0.0;    // the same on the CPU
    std::vector<drive_profile> drives;

    bool low_power_mode = false; // macOS Low Power Mode cuts writing speed (spec 12.1 item 10)
    std::vector<std::string> notes; // plain-words warnings found during setup

    nlohmann::json to_json() const;
    static machine_profile from_json(const nlohmann::json & j);
};

// Steps of auto-setup reported to the app as they happen (0..1 overall progress).
using setup_progress = std::function<void(double fraction, const std::string & step)>;

struct setup_options {
    bool quick = false;          // small tests (CI, unit tests): seconds instead of about a minute
    bool measure_disks = true;
    bool measure_gpu = true;
    std::vector<std::string> only_drives; // empty = every drive with enough free space
    setup_progress progress;
};

// Section 4a steps 1-2: detect and measure.
machine_profile run_setup(const setup_options & opt);

// Step 3: explain in plain words, one sentence per line.
std::vector<std::string> explain(const machine_profile & p);

// Step 4: save and load the profile (kera_machine.json in the Kera folder by default).
std::string profile_path();
bool save_profile(const machine_profile & p, const std::string & path);
bool load_profile(machine_profile & p, const std::string & path);

// Detection pieces (detect.cpp), exposed for tests.
void detect_system(machine_profile & p);
// Memory the OS can hand out now without paging anything out, in GB (0 if unknown)
double available_ram_gb();
std::vector<drive_profile> list_drives();

// Measurement pieces.
void measure_drive(drive_profile & d, const std::string & test_dir, const setup_options & opt);
// Read an existing file at random offsets, cache bypassed: GB/s. For cross-checks against fio and similar tools.
double read_test_file(const std::string & path, int read_mib, int inflight, double seconds, int * error);
double measure_cpu_read_gbps(size_t bytes, int threads, double seconds);
void measure_gpus(machine_profile & p, const setup_options & opt);
// an NVIDIA chip that shares the computer's memory (GB10, Jetson), which llama.cpp reports as a separate card
bool nvidia_shares_memory(const std::string & backend, const std::string & name, double total_gb, double ram_gb);

// Formatting helpers used by explain() and the app.
std::string fmt_gbps(double gbps);
std::string drive_label(const drive_profile & d); // "the main drive", "drive D:", "/Volumes/Fast SSD"
std::string utc_now_iso8601();

} // namespace kera
