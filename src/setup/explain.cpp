// Auto-setup step 3 (spec 4a): explain the measurements in plain words, without jargon.

#include "setup/kr_machine.h"

#include <cmath>
#include <cstdio>

namespace kera {

std::string fmt_gbps(double gbps) {
    char buf[32];
    if (gbps >= 100) {
        std::snprintf(buf, sizeof(buf), "%.0f GB/s", gbps);
    } else if (gbps >= 10) {
        std::snprintf(buf, sizeof(buf), "%.0f GB/s", gbps);
    } else {
        std::snprintf(buf, sizeof(buf), "%.1f GB/s", gbps);
    }
    return buf;
}

std::string drive_label(const drive_profile & d) {
    const std::string & p = d.path;
    if (p == "/" || p == "/System/Volumes/Data") {
        return "the main drive";
    }
    if (p.size() >= 2 && p[1] == ':') {
        return "the " + p.substr(0, 2) + " drive"; // "C:\\" -> "the C: drive"
    }
    return p;
}

static std::string fmt_gb(double gb) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), gb >= 10 ? "%.0f GB" : "%.1f GB", gb);
    return buf;
}

static std::string cap(std::string s) {
    if (!s.empty() && s[0] >= 'a' && s[0] <= 'z') {
        s[0] = (char) (s[0] - 'a' + 'A');
    }
    return s;
}

std::vector<std::string> explain(const machine_profile & p) {
    std::vector<std::string> out;

    std::string chip = p.chip.empty() ? "an unknown processor" : p.chip;
    std::string line = "This computer has " + chip;
    if (p.cpu_cores > 0) {
        line += " with " + std::to_string(p.cpu_cores) + (p.cpu_cores == 1 ? " core" : " cores");
    }
    line += ", " + fmt_gb(p.ram_gb) + " of memory (RAM), and runs " + p.os + ".";
    out.push_back(line);

    if (!p.gpus.empty()) {
        const gpu_profile & g = p.gpus[0];
        std::string l = "Its graphics chip is " + g.name + " (run through " + g.backend + ")";
        if (g.unified) {
            l += ". It shares the computer's memory with the CPU, which suits big models";
        }
        out.push_back(l + ".");
    } else {
        out.push_back("Kera didn't find a graphics chip it can use, so models will run on the CPU.");
    }

    out.push_back("Kera expects a model can use about " + fmt_gb(p.usable_gb) +
                  " of that memory. Models bigger than that are read from the SSD while they run.");

    // The memory split (spec Phase 8 items 3 and 6): on Windows, a graphics chip that shares the computer's memory
    // may only use the part set aside for it. When that is much less than the memory there is, say how to change it.
    if (!p.gpus.empty() && p.gpus[0].unified && p.os_family == "windows" && p.gpus[0].total_gb > 0 &&
        p.gpus[0].total_gb < 0.6 * p.ram_gb) {
        const gpu_profile & g = p.gpus[0];
        out.push_back("Windows lets the graphics chip use " + fmt_gb(g.total_gb) + " of the " + fmt_gb(p.ram_gb) +
                      ". For models bigger than that, giving it more lets more of the model stay in memory: in AMD "
                      "Software, Performance, Tuning, raise Variable Graphics Memory (or the UMA frame buffer size in "
                      "the computer's BIOS), then restart. Kera never changes this by itself.");
    }

    if (p.ram_gbps.gpu > 0) {
        out.push_back("The graphics chip reads memory at " + fmt_gbps(p.ram_gbps.gpu) +
                      ". That sets the top writing speed for models that fit in memory.");
    }
    if (p.ram_gbps.cpu > 0) {
        out.push_back("The CPU reads memory at " + fmt_gbps(p.ram_gbps.cpu) + " using all its cores.");
    }
    if (p.gpu_q4_gbps > 0) {
        char pct[16];
        std::snprintf(pct, sizeof(pct), "%.0f%%", p.ram_gbps.gpu > 0 ? 100.0 * p.gpu_q4_gbps / p.ram_gbps.gpu : 0.0);
        out.push_back("On a 4-bit model layer, the graphics chip worked through " + fmt_gbps(p.gpu_q4_gbps) +
                      " of weights" + (p.ram_gbps.gpu > 0 ? std::string(" (") + pct + " of its memory speed)" : "") + ".");
    } else if (p.cpu_q4_gbps > 0) {
        out.push_back("On a 4-bit model layer, the CPU worked through " + fmt_gbps(p.cpu_q4_gbps) + " of weights.");
    }

    for (const auto & d : p.drives) {
        if (d.measured) {
            std::string l = cap(drive_label(d)) + " reads " + fmt_gbps(d.read_gbps);
            if (d.best_inflight > 1) {
                l += " when Kera reads " + std::to_string(d.best_inflight) + " pieces of " +
                     std::to_string(d.best_read_mib) + " MB at once";
                if (d.single_read_gbps > 0) {
                    l += " (" + fmt_gbps(d.single_read_gbps) + " one piece at a time, which is what simple tests show)";
                }
            } else {
                l += ", reading " + std::to_string(d.best_read_mib) + " MB pieces one at a time";
            }
            out.push_back(l + ".");
        } else if (!d.note.empty()) {
            out.push_back("Kera didn't measure " + drive_label(d) + ". " + d.note);
        }
        if (d.measured && !d.note.empty()) {
            out.push_back(d.note);
        }
    }

    if (p.low_power_mode) {
        out.push_back("Low Power Mode is on. It can cut writing speed by about a third; turn it off in Settings, Battery for full speed.");
    }
    for (const auto & n : p.notes) {
        out.push_back(n);
    }
    return out;
}

} // namespace kera
