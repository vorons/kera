// What Kera learned about each guesser per model (kr_guess_memory.h).

#include "spec/kr_guess_memory.h"

#include <chrono>
#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace kera {

guess_memory::guess_memory(std::string path) : path_(std::move(path)) {
    std::ifstream f(fs::u8path(path_));
    if (f) {
        try {
            j_ = nlohmann::json::parse(f);
        } catch (const std::exception &) {
            j_ = nlohmann::json::object(); // a damaged file only loses what was learned
        }
    }
    if (!j_.is_object()) {
        j_ = nlohmann::json::object();
    }
}

bool guess_memory::get(const std::string & model, const std::string & guesser, guess_verdict & out, int max_age_days) const {
    std::lock_guard<std::mutex> lock(mu_);
    if (!j_.contains(model) || !j_[model].is_object() || !j_[model].contains(guesser)) {
        return false;
    }
    const auto & v = j_[model][guesser];
    guess_verdict r;
    r.pays = v.value("pays", true);
    r.time = v.value("time", (int64_t) 0);
    r.steps = v.value("steps", 0LL);
    r.words_per_s = v.value("words_per_s", 0.0);
    const int64_t now = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch()).count();
    if (now - r.time > (int64_t) max_age_days * 86400) {
        return false;
    }
    out = r;
    return true;
}

void guess_memory::set(const std::string & model, const std::string & guesser, const guess_verdict & v) {
    std::lock_guard<std::mutex> lock(mu_);
    j_[model][guesser] = { { "pays", v.pays }, { "time", v.time }, { "steps", v.steps }, { "words_per_s", v.words_per_s } };
    const std::string tmp = path_ + ".tmp";
    {
        std::ofstream f(fs::u8path(tmp), std::ios::trunc);
        f << j_.dump(1) << "\n";
        if (!f) {
            return;
        }
    }
    std::error_code ec;
    fs::rename(fs::u8path(tmp), fs::u8path(path_), ec);
}

std::string guess_memory_key(const std::string & model_path) {
    std::error_code ec;
    const fs::path p = fs::u8path(model_path);
    const auto size = fs::file_size(p, ec);
    return p.filename().u8string() + "|" + std::to_string(ec ? 0 : (unsigned long long) size);
}

} // namespace kera
