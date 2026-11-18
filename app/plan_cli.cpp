// `kera plan --model FILE.gguf`: what goes in memory and on the SSD, the GB read per word, and the
// speed ceiling (spec Phase 1 step 4), from the model's header and the saved machine profile.

#include "app/kera_app.h"
#include "planner/kr_plan.h"
#include "setup/kr_machine.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

namespace kera {

int run_plan_cli(int argc, char ** argv) {
    std::string model;
    bool as_json = false;
    double hit = -1, usable = -1, ram = -1, ssd = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() { return i + 1 < argc ? std::atof(argv[++i]) : 0.0; };
        if ((a == "--model" || a == "-m") && i + 1 < argc) {
            model = argv[++i];
        } else if (a == "--hit-rate") {
            hit = val();
        } else if (a == "--usable-gb") {
            usable = val();
        } else if (a == "--ram-gbps") {
            ram = val();
        } else if (a == "--ssd-gbps") {
            ssd = val();
        } else if (a == "--json") {
            as_json = true;
        } else {
            model = a; // `kera plan FILE.gguf`
        }
    }
    if (model.empty()) {
        std::fprintf(stderr, "Usage: kera plan --model FILE.gguf [--hit-rate 0..1] [--usable-gb N] [--ram-gbps N] [--ssd-gbps N] [--json]\n"
                             "Uses the machine profile from `kera setup`; the options override its numbers.\n");
        return 2;
    }
    model_sizes m;
    std::string err;
    if (!read_model_sizes(model, m, &err)) {
        std::fprintf(stderr, "Can't read %s: %s\n", model.c_str(), err.c_str());
        return 1;
    }
    machine_profile p;
    const bool have_profile = load_profile(p, profile_path());
    plan_input in = have_profile ? plan_input_from_profile(p) : plan_input{};
    if (usable >= 0) in.usable_ram_gb = usable;
    if (ram >= 0) in.ram_gb_s = ram;
    if (ssd >= 0) in.ssd_gb_s = ssd;
    in.hit_rate = hit;
    if (in.usable_ram_gb <= 0 || in.ram_gb_s <= 0) {
        std::fprintf(stderr, "No machine profile yet: run `kera setup` first, or give --usable-gb and --ram-gbps.\n");
        return 1;
    }
    const plan_result r = plan(m, in);
    if (as_json) {
        const nlohmann::json j = {
            { "model", { { "file", m.file }, { "arch", m.arch }, { "file_gb", m.file_gb }, { "always_gb", m.always_gb },
                         { "embed_gb", m.embed_gb }, { "experts_gb", m.experts_gb }, { "active_experts_gb", m.active_experts_gb },
                         { "n_expert", m.n_expert }, { "n_expert_used", m.n_expert_used }, { "n_moe_layers", m.n_moe_layers } } },
            { "machine", { { "usable_ram_gb", in.usable_ram_gb }, { "ram_gb_s", in.ram_gb_s }, { "ssd_gb_s", in.ssd_gb_s } } },
            { "plan", { { "fits", r.fits }, { "in_memory_gb", r.in_memory_gb }, { "on_ssd_gb", r.on_ssd_gb },
                        { "expert_cache_gb", r.expert_cache_gb }, { "hit_rate", r.hit_rate },
                        { "ram_gb_per_word", r.ram_gb_per_word }, { "ssd_gb_per_word", r.ssd_gb_per_word },
                        { "ceiling_tok_s", r.ceiling_tok_s }, { "ceiling_overlap_tok_s", r.ceiling_overlap_tok_s } } },
        };
        std::printf("%s\n", j.dump(2).c_str());
        return 0;
    }
    for (const auto & line : explain_plan(m, in, r)) {
        std::printf("%s\n", line.c_str());
    }
    std::printf("\n  in memory      %8.2f GB\n  on the SSD     %8.2f GB\n  read per word  %8.3f GB from memory, %.3f GB from the SSD\n"
                "  ceiling        %8.1f tok/s (%.1f tok/s with SSD reads overlapping the maths)\n",
                r.in_memory_gb, r.on_ssd_gb, r.ram_gb_per_word, r.ssd_gb_per_word, r.ceiling_tok_s, r.ceiling_overlap_tok_s);
    return 0;
}

} // namespace kera
