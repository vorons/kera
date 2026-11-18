// Machine profile JSON (spec P3): kera_machine.json, read by the planner and the tuner.

#include "setup/kr_machine.h"

#include "common/kr_paths.h"

#include <ctime>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace kera {

using nlohmann::json;

static double round2(double v) {
    return std::round(v * 100.0) / 100.0;
}

json machine_profile::to_json() const {
    json gpus_j = json::array();
    for (const auto & g : gpus) {
        gpus_j.push_back({
            { "name", g.name }, { "backend", g.backend }, { "total_gb", round2(g.total_gb) },
            { "free_gb", round2(g.free_gb) }, { "max_alloc_gb", round2(g.max_alloc_gb) }, { "unified", g.unified },
        });
    }
    json drives_j = json::array();
    for (const auto & d : drives) {
        drives_j.push_back({
            { "path", d.path }, { "label", drive_label(d) }, { "device", d.device }, { "total_gb", round2(d.total_gb) }, { "free_gb", round2(d.free_gb) },
            { "measured", d.measured }, { "read_gbps", round2(d.read_gbps) }, { "write_gbps", round2(d.write_gbps) },
            { "best_read_mib", d.best_read_mib }, { "best_inflight", d.best_inflight },
            { "single_read_gbps", round2(d.single_read_gbps) }, { "direct", d.direct }, { "note", d.note },
        });
    }
    // P3 shape: gpu{name, max_alloc_gb} is the main graphics chip; "gpus" lists all of them
    json gpu_j = gpus.empty() ? json(nullptr)
                              : json{ { "name", gpus[0].name }, { "max_alloc_gb", round2(gpus[0].max_alloc_gb) } };
    return {
        { "kera_version", kera_version },
        { "measured_at", measured_at },
        { "os", os },
        { "os_family", os_family },
        { "chip", chip },
        { "cpu_cores", cpu_cores },
        { "cpu_threads", cpu_threads },
        { "ram_gb", round2(ram_gb) },
        { "usable_gb", round2(usable_gb) },
        { "gpu", gpu_j },
        { "gpus", gpus_j },
        { "ram_gbps", { { "gpu", round2(ram_gbps.gpu) }, { "cpu", round2(ram_gbps.cpu) }, { "both", round2(ram_gbps.both) } } },
        { "gpu_q4_gbps", round2(gpu_q4_gbps) },
        { "cpu_q4_gbps", round2(cpu_q4_gbps) },
        { "drives", drives_j },
        { "low_power_mode", low_power_mode },
        { "notes", notes },
    };
}

machine_profile machine_profile::from_json(const json & j) {
    machine_profile p;
    p.kera_version = j.value("kera_version", "");
    p.measured_at  = j.value("measured_at", "");
    p.os           = j.value("os", "");
    p.os_family    = j.value("os_family", "");
    p.chip         = j.value("chip", "");
    p.cpu_cores    = j.value("cpu_cores", 0);
    p.cpu_threads  = j.value("cpu_threads", 0);
    p.ram_gb       = j.value("ram_gb", 0.0);
    p.usable_gb    = j.value("usable_gb", 0.0);
    if (j.contains("gpus") && j["gpus"].is_array()) {
        for (const auto & g : j["gpus"]) {
            gpu_profile gp;
            gp.name = g.value("name", "");
            gp.backend = g.value("backend", "");
            gp.total_gb = g.value("total_gb", 0.0);
            gp.free_gb = g.value("free_gb", 0.0);
            gp.max_alloc_gb = g.value("max_alloc_gb", 0.0);
            gp.unified = g.value("unified", false);
            p.gpus.push_back(gp);
        }
    }
    if (j.contains("ram_gbps")) {
        p.ram_gbps.gpu = j["ram_gbps"].value("gpu", 0.0);
        p.ram_gbps.cpu = j["ram_gbps"].value("cpu", 0.0);
        p.ram_gbps.both = j["ram_gbps"].value("both", 0.0);
    }
    p.gpu_q4_gbps = j.value("gpu_q4_gbps", 0.0);
    p.cpu_q4_gbps = j.value("cpu_q4_gbps", 0.0);
    if (j.contains("drives") && j["drives"].is_array()) {
        for (const auto & d : j["drives"]) {
            drive_profile dp;
            dp.path = d.value("path", "");
            dp.device = d.value("device", "");
            dp.total_gb = d.value("total_gb", 0.0);
            dp.free_gb = d.value("free_gb", 0.0);
            dp.measured = d.value("measured", false);
            dp.read_gbps = d.value("read_gbps", 0.0);
            dp.write_gbps = d.value("write_gbps", 0.0);
            dp.best_read_mib = d.value("best_read_mib", 0);
            dp.best_inflight = d.value("best_inflight", 0);
            dp.single_read_gbps = d.value("single_read_gbps", 0.0);
            dp.direct = d.value("direct", false);
            dp.note = d.value("note", "");
            p.drives.push_back(dp);
        }
    }
    p.low_power_mode = j.value("low_power_mode", false);
    if (j.contains("notes") && j["notes"].is_array()) {
        for (const auto & n : j["notes"]) {
            p.notes.push_back(n.get<std::string>());
        }
    }
    return p;
}

std::string profile_path() {
    return home_path("kera_machine.json");
}

bool save_profile(const machine_profile & p, const std::string & path) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(fs::u8path(tmp), std::ios::binary);
        if (!f) {
            return false;
        }
        f << p.to_json().dump(2) << "\n";
        if (!f) {
            return false;
        }
    }
    // (UTF-8 paths through the filesystem library: Windows reads narrow ones in the ANSI code page)
    std::error_code ec;
    fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
    return !ec;
}

bool load_profile(machine_profile & p, const std::string & path) {
    std::ifstream f(fs::u8path(path), std::ios::binary);
    if (!f) {
        return false;
    }
    try {
        p = machine_profile::from_json(json::parse(f));
        return true;
    } catch (const std::exception &) {
        return false;
    }
}

std::string utc_now_iso8601() {
    const std::time_t t = std::time(nullptr);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

} // namespace kera
