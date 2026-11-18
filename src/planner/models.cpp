// Finding model files (kr_models.h).

#include "planner/kr_models.h"

#include "common/kr_paths.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>

namespace fs = std::filesystem;

namespace kera {

namespace {

std::string env_or(const char * name, const std::string & def) {
    const char * v = std::getenv(name);
    return v && *v ? std::string(v) : def;
}

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// "name-00002-of-00004.gguf": a later part of a split model (the first part stands for all of them)
bool later_split_part(const std::string & name) {
    const size_t of = name.rfind("-of-");
    if (of == std::string::npos || of < 6) {
        return false;
    }
    const std::string part = name.substr(of - 5, 5);
    return std::all_of(part.begin(), part.end(), ::isdigit) && part != "00001";
}

} // namespace

std::vector<std::string> model_folders() {
    std::vector<std::string> out = { home_path("models") };
    std::ifstream f(fs::u8path(home_path("model_folders.json")));
    if (f) {
        try {
            for (const auto & v : nlohmann::json::parse(f)) {
                out.push_back(v.get<std::string>());
            }
        } catch (const std::exception &) {
        }
    }
#if defined(_WIN32)
    const std::string home = env_or("USERPROFILE", "");
    const std::string local = env_or("LOCALAPPDATA", "");
#else
    const std::string home = env_or("HOME", "");
    const std::string local = env_or("XDG_CACHE_HOME", home.empty() ? "" : join(home, ".cache"));
#endif
    if (!home.empty()) {
        out.push_back(join(join(home, ".lmstudio"), "models"));
        out.push_back(join(join(join(home, ".cache"), "lm-studio"), "models"));
        out.push_back(join(join(join(home, ".cache"), "huggingface"), "hub"));
    }
    if (!local.empty()) {
        out.push_back(join(local, "llama.cpp"));
    }
    return out;
}

bool add_model_folder(const std::string & folder) {
    std::error_code ec;
    if (!fs::is_directory(fs::u8path(folder), ec)) {
        return false;
    }
    nlohmann::json list = nlohmann::json::array();
    {
        std::ifstream f(fs::u8path(home_path("model_folders.json")));
        if (f) {
            try {
                list = nlohmann::json::parse(f);
            } catch (const std::exception &) {
                list = nlohmann::json::array();
            }
        }
    }
    for (const auto & v : list) {
        if (v == folder) {
            return true;
        }
    }
    list.push_back(folder);
    std::ofstream(fs::u8path(home_path("model_folders.json"))) << list.dump(1) << "\n";
    return true;
}

std::vector<std::string> find_models(const std::vector<std::string> & folders) {
    std::vector<std::string> out;
    std::set<std::string> seen;
    for (const auto & folder : folders) {
        std::error_code ec;
        const fs::path root = fs::u8path(folder);
        if (!fs::is_directory(root, ec)) {
            continue;
        }
        fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
        for (; !ec && it != end; it.increment(ec)) {
            if (it.depth() > 6) {
                it.disable_recursion_pending();
                continue;
            }
            if (!it->is_regular_file(ec)) {
                continue;
            }
            const std::string name = lower(it->path().filename().u8string());
            if (name.size() < 5 || name.substr(name.size() - 5) != ".gguf" || name.find("mmproj") != std::string::npos ||
                later_split_part(name)) {
                continue;
            }
            // Hugging Face's cache links snapshot files to blobs: list each real file once
            const std::string real = fs::weakly_canonical(it->path(), ec).u8string();
            if (seen.insert(real.empty() ? it->path().u8string() : real).second) {
                out.push_back(fs::absolute(it->path(), ec).u8string());
            }
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

} // namespace kera
