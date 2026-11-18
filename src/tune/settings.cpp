// Settings registry (spec P3, P1 item 2). Thread-safe: settings are registered at start-up and read
// from any thread.

#include "tune/kr_settings.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct setting {
    std::string name;
    std::string help;
    double def;
    double min;
    double max;
    double value;
    int tunable;
};

struct registry {
    std::mutex mutex;
    std::vector<setting> items;
    std::unordered_map<std::string, size_t> index;
};

registry & reg() {
    static registry r;
    return r;
}

double clamp(double v, double lo, double hi) {
    return std::min(std::max(v, lo), hi);
}

} // namespace

extern "C" int kr_setting_register(const char * name, double def, double min, double max, int tunable, const char * help) {
    if (name == nullptr || !(min <= max)) {
        return 0;
    }
    registry & r = reg();
    std::lock_guard<std::mutex> lock(r.mutex);
    if (r.index.count(name)) {
        return 0;
    }
    const double d = clamp(def, min, max);
    r.index[name] = r.items.size();
    r.items.push_back({ name, help ? help : "", d, min, max, d, tunable });
    return 1;
}

extern "C" double kr_setting_get(const char * name) {
    registry & r = reg();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = name ? r.index.find(name) : r.index.end();
    if (it == r.index.end()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return r.items[it->second].value;
}

extern "C" int kr_setting_set(const char * name, double value) {
    registry & r = reg();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = name ? r.index.find(name) : r.index.end();
    if (it == r.index.end() || std::isnan(value)) {
        return 0;
    }
    setting & s = r.items[it->second];
    s.value = clamp(value, s.min, s.max);
    return 1;
}

extern "C" void kr_setting_reset(const char * name) {
    registry & r = reg();
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = name ? r.index.find(name) : r.index.end();
    if (it != r.index.end()) {
        setting & s = r.items[it->second];
        s.value = s.def;
    }
}

extern "C" size_t kr_settings_count(void) {
    registry & r = reg();
    std::lock_guard<std::mutex> lock(r.mutex);
    return r.items.size();
}

extern "C" int kr_settings_get_info(size_t i, kr_setting_info * out) {
    registry & r = reg();
    std::lock_guard<std::mutex> lock(r.mutex);
    if (out == nullptr || i >= r.items.size()) {
        return 0;
    }
    const setting & s = r.items[i];
    // the strings live as long as the registry, which is never cleared
    *out = { s.name.c_str(), s.def, s.min, s.max, s.value, s.tunable, s.help.c_str() };
    return 1;
}

extern "C" void kr_settings_register_builtin(void) {
    // Auto-setup (section 4a). Not tunable: they decide how long the machine check takes.
    kr_setting_register("setup.ssd_test_gb",       2.0, 0.25, 4.0, 0, "Size of the SSD speed-test file in GB (spec: 2-4 GB)");
    kr_setting_register("setup.ssd_cell_seconds",  0.8, 0.1,  5.0, 0, "Seconds spent on each read size and thread count in the SSD test");
    kr_setting_register("setup.ssd_confirm_seconds", 3.0, 0.2, 30.0, 0, "Seconds spent re-measuring the best SSD read setting, which is the speed reported");
    kr_setting_register("setup.ram_test_mib",   1024.0, 64.0, 8192.0, 0, "Size of the RAM speed-test buffer in MiB");

    // I/O (section 4). 0 = take the best combination auto-setup measured for that drive.
    kr_setting_register("io.reader_threads", 0.0, 0.0, 64.0, 1, "Reads in flight for weights (0 = measured best)");
    kr_setting_register("io.read_size_mib",  0.0, 0.0, 64.0, 1, "Size of each weight read in MiB (0 = measured best)");
    kr_setting_register("kv.store",          1.0, 0.0, 1.0, 0, "Save chat states to the SSD, so old chats and restarts skip re-reading (1 = on)");
    kr_setting_register("kv.disk_budget_gb", 20.0, 1.0, 2000.0, 0, "Most SSD space saved chat states may use, in GB (least recently used go first)");
    kr_setting_register("spec.controller",   1.0, 0.0, 1.0, 0, "Pick each step's number of guessed words by measured speed, 0 included (1 = on)");
    kr_setting_register("spec.auto",         1.0, 0.0, 1.0, 0, "When nothing else is set, guess by copying from the chat, and with the model's own MTP head if it has one (1 = on)");
    kr_setting_register("chat.model_defaults", 1.0, 0.0, 1.0, 0, "Use the sampling the model's card recommends for thinking, no thinking and tool calls (1 = on)");
    kr_setting_register("gpu.shader_cache",  1.0, 0.0, 1.0, 0, "Keep the shaders the graphics chip builds in Kera's folder, so a new Kera folder or an update starts warm (Vulkan; 1 = on)");
    kr_setting_register("image.warmup",      1.0, 0.0, 1.0, 0, "When an image model is served on a graphics chip, make one small throwaway picture as it starts, so the first real one is as fast as the next (the owner's request, D-032; 1 = on)");
    kr_setting_register("gpu.shared_no_mmap", 1.0, 0.0, 1.0, 0, "On an NVIDIA chip that shares the computer's memory (DGX Spark, GB10), load the model into memory instead of memory-mapping it (spec 10.6: 56.3 against 44.7 tok/s measured on a Spark; 1 = on)");
    kr_setting_register("gpu.auto_backend",  0.0, 0.0, 1.0, 0, "With the ROCm add-on installed and no --backend, measure Vulkan and ROCm once per model and use the faster for how the model is used (1 = on; --backend auto does it once)");
    kr_setting_register("serve.short_first",  0.0, 0.0, 1.0, 0, "With several chats at once, read the prompt with the fewest tokens left first, so a new message goes before the rest of a long one (1 = on; --multi-user)");
    kr_setting_register("serve.read_share",   0.0, 0.0, 1.0, 1, "With several chats at once, the most of the chip's time reading a long prompt takes while others write (0 = no limit, as llama.cpp; --multi-user: 0.5)");
    kr_setting_register("serve.read_step_s",  0.5, 0.05, 10.0, 1, "While others write, about the longest one step of prompt reading takes, in seconds");
    kr_setting_register("serve.share_prefix", 0.0, 0.0, 1.0, 0, "Let a chat share the start it has in common with another chat's memory instead of reading it again (1 = on; plain attention models; --multi-user)");
    kr_setting_register("serve.metrics",      0.0, 0.0, 1.0, 0, "Serve llama.cpp's load and speed counters at /metrics (1 = on; --multi-user)");
    kr_setting_register("stream.mode",       -1.0, -1.0, 2.0, 1, "Stream a MoE model's experts from the SSD: -1 when it doesn't fit, 0 never, 1 with the OS file cache, 2 with Kera's cache");
    kr_setting_register("stream.budget_gb",  0.0, 0.0, 4096.0, 1, "RAM for Kera's expert cache in GB (0 = what the machine profile leaves after the rest of the model)");
    kr_setting_register("stream.lookahead",  0.0, 0.0, 1.0, 1, "Read the next MoE layer's likely experts early (Phase 3; on only once it pays on this machine)");
    kr_setting_register("stream.io_threads", 8.0, 1.0, 64.0, 1, "Expert reads in flight");
    // (Kera has no settings that trade quality for speed: its lossy modes were removed, D-034)
}
