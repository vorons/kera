// The same model at a higher precision next to the file the user named (D-036): Kera uses the file it was given, and
// says when the model's own weights (or a higher-precision copy) are beside it and fit this computer's memory.

#include "app/kera_app.h"
#include "setup/kr_machine.h"
#include "tune/kr_precision.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>

namespace fs = std::filesystem;

namespace kera {

double gpu_memory_gb() {
    ggml_backend_load_all();
    double best = 0.0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(ggml_backend_dev_get(i), &props);
        if (props.type == GGML_BACKEND_DEVICE_TYPE_GPU || props.type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            best = std::max(best, (double) props.memory_total / 1e9);
        }
    }
    return best;
}

void note_higher_precision(const std::string & model, const char * option) {
    if (model.empty()) {
        return;
    }
    const fs::path path = fs::u8path(model);
    const std::string name = path.filename().u8string();
    if (precision_of(name) <= 1) {
        return; // the model's own weights already, or a name that doesn't say
    }
    std::vector<sibling_file> folder;
    std::error_code ec;
    const fs::path dir = path.has_parent_path() ? path.parent_path() : fs::path(".");
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code size_ec;
        const auto bytes = it->is_regular_file(size_ec) ? fs::file_size(it->path(), size_ec) : 0;
        if (!size_ec && bytes > 0) {
            folder.push_back({ it->path().filename().u8string(), (uint64_t) bytes });
        }
    }
    if (higher_precision_sibling(name, folder, 0).empty()) {
        return; // nothing beside it: the usual case, with no chip looked at
    }
    machine_profile p;
    detect_system(p);
    const double gpu = gpu_memory_gb();
    const double memory = gpu > 0 ? gpu : p.ram_gb;
    const std::string better = higher_precision_sibling(name, folder, (uint64_t) (0.75 * memory * 1e9));
    if (better.empty()) {
        return;
    }
    const bool full = precision_of(better) <= 1;
    std::fprintf(stderr, "Kera: %s is next to this file: the same model at %s, and it fits this computer's memory "
                         "(%s %s uses it)\n", better.c_str(),
                 full ? "full precision (the model's own weights)" : "a higher precision", option,
                 (dir / fs::u8path(better)).u8string().c_str());
}

} // namespace kera
