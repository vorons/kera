// `kera setup`: auto-setup from the command line (spec 4a, Phase 13 command list).

#include "app/kera_app.h"

#include "io/kr_io.h"
#include "setup/kr_machine.h"
#include "tune/kr_settings.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace kera {

int run_setup_cli(int argc, char ** argv) {
    setup_options opt;
    bool json = false;
    std::string out = profile_path();
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--quick") {
            opt.quick = true;
        } else if (a == "--json") {
            json = true;
        } else if (a == "--no-disk") {
            opt.measure_disks = false;
        } else if (a == "--no-gpu") {
            opt.measure_gpu = false;
        } else if (a == "--drive" && i + 1 < argc) {
            opt.only_drives.push_back(argv[++i]);
        } else if (a == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else if (a == "--set" && i + 1 < argc) {
            // --set name=value overrides a setting, e.g. --set setup.ssd_confirm_seconds=5
            kr_settings_register_builtin();
            const std::string kv = argv[++i];
            const size_t eq = kv.find('=');
            if (eq == std::string::npos || !kr_setting_set(kv.substr(0, eq).c_str(), std::atof(kv.c_str() + eq + 1))) {
                std::fprintf(stderr, "Unknown setting in --set %s\n", kv.c_str());
                return 1;
            }
        } else if (a == "--help" || a == "-h") {
            std::printf(
                "Usage: kera setup [--quick] [--json] [--no-disk] [--no-gpu] [--drive PATH] [--out FILE] [--set NAME=VALUE]\n"
                "Measures this computer (SSD, memory, graphics chip), explains the results, and saves them.\n"
                "  --quick      small tests that take seconds (less accurate)\n"
                "  --json       print the machine profile as JSON\n"
                "  --drive PATH measure only this drive (can be repeated)\n"
                "  --out FILE   where to save the profile (default: %s)\n"
                "  --set N=V    override a setting, e.g. --set setup.ssd_confirm_seconds=5\n", profile_path().c_str());
            return 0;
        }
    }
    if (!json) {
        opt.progress = [](double f, const std::string & step) {
            std::fprintf(stderr, "[%3.0f%%] %s\n", 100.0 * f, step.c_str());
        };
    }

    const machine_profile p = run_setup(opt);
    const bool saved = save_profile(p, out);

    if (json) {
        std::printf("%s\n", p.to_json().dump(2).c_str());
    } else {
        std::printf("\n");
        for (const auto & line : explain(p)) {
            std::printf("- %s\n", line.c_str());
        }
        std::printf("\nI/O method: %s\n", kr_io_backend_name());
        std::printf(saved ? "Saved to %s\n" : "Couldn't save to %s\n", out.c_str());
    }
    return saved ? 0 : 1;
}

} // namespace kera
