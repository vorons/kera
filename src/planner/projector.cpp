// Finding a vision model's image reader, and Ollama's models (projector.h).

#include "planner/projector.h"

#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>

namespace fs = std::filesystem;

namespace kera {

namespace {

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// a name without its extension, "mmproj", the model-type words and separators, for comparing two files' names
// a file's name without what doesn't say which model it is: the folder, ".gguf", the split part ("-00001-of-00003"),
// the number format ("Q4_K_M", "f16", "BF16", "IQ3_XXS"), the words "mmproj" and "model", and separators.
// "mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf" and "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf" both give "qwen25vl7binstruct"
std::string stem_key(const std::string & file) {
    std::string s = lower(fs::u8path(file).stem().u8string());
    static const std::regex split("[-_.]?\\d{5}-of-\\d{5}$");
    static const std::regex format("([-_.](i?q[0-9]+(_[0-9a-z]+)*|bf16|fp16|fp32|f16|f32|mxfp4(_moe)?))+$");
    s = std::regex_replace(s, split, "");
    s = std::regex_replace(s, format, "");
    for (const char * w : { "mmproj", "model" }) {
        for (size_t at; (at = s.find(w)) != std::string::npos;) {
            s.erase(at, std::strlen(w));
        }
    }
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '-' || c == '_' || c == '.' || c == ' '; }),
            s.end());
    return s;
}

// a reader made for this model: the names agree up to the end of the shorter one ("gemma34b" for "gemma34bit"), so
// sizes, versions and variants must match ("qwen25vl3b" isn't "qwen25vl7b"'s, "qwen257b" isn't "qwen25vl7b"'s)
bool names_match(const std::string & reader, const std::string & model) {
    if (reader.size() < 3 || model.size() < 3) {
        return false; // "mmproj-F16.gguf" names no model: only the one-model-one-reader rule can pick it
    }
    const std::string & shorter = reader.size() < model.size() ? reader : model;
    const std::string & longer = reader.size() < model.size() ? model : reader;
    if (longer.compare(0, shorter.size(), shorter) != 0) {
        return false;
    }
    // where the longer name goes on to say a size ("gemma3" + "12bit"), the shorter must say one too: "gemma3"'s
    // reader may be for another size of it
    static const std::regex size("[0-9]+[bm]");
    return !std::regex_search(longer.substr(shorter.size()), size) || std::regex_search(shorter, size);
}

// a file's number format in bits, from its name's last part as llama.cpp reads it ("...-Q8_0" 8, "...-bf16" 16,
// "...-F32" 32); -1 if the name gives none
int quant_bits(const std::string & file) {
    const std::string stem = fs::u8path(file).stem().u8string();
    const size_t at = stem.find_last_of("-.");
    if (at == std::string::npos) {
        return -1;
    }
    const std::string tag = stem.substr(at + 1);
    const size_t d = tag.find_first_of("0123456789");
    if (d == std::string::npos || tag.empty() || !std::isalpha((unsigned char) tag[0])) {
        return -1;
    }
    return std::atoi(tag.c_str() + d);
}

size_t common_prefix(const std::string & a, const std::string & b) {
    size_t n = 0;
    while (n < a.size() && n < b.size() && a[n] == b[n]) {
        ++n;
    }
    return n;
}

fs::path blob_path(const fs::path & models, const std::string & digest) {
    std::string d = digest;
    std::replace(d.begin(), d.end(), ':', '-'); // "sha256:ab..." is stored as blobs/sha256-ab...
    return models / "blobs" / fs::u8path(d);
}

fs::path ollama_folder(const std::string & folder) {
    if (!folder.empty()) {
        return fs::u8path(folder);
    }
    if (const char * e = std::getenv("OLLAMA_MODELS"); e != nullptr && *e != '\0') {
        return fs::u8path(e);
    }
#if defined(_WIN32)
    const char * home = std::getenv("USERPROFILE");
#else
    const char * home = std::getenv("HOME");
#endif
    return home ? fs::u8path(home) / ".ollama" / "models" : fs::path();
}

// what one manifest file names: the model's blob, its projector's, and the rest of its Modelfile ("" where it names none)
void read_manifest(const fs::path & file, const fs::path & models, ollama_model & m) {
    m = ollama_model{};
    std::ifstream f(file);
    const nlohmann::json j = nlohmann::json::parse(f, nullptr, false);
    if (!j.is_object() || !j.contains("layers") || !j["layers"].is_array()) {
        return;
    }
    for (const auto & l : j["layers"]) {
        if (!l.is_object()) {
            continue;
        }
        const std::string type = l.value("mediaType", "");
        const std::string digest = l.value("digest", "");
        if (digest.empty()) {
            continue;
        }
        const std::string blob = blob_path(models, digest).u8string();
        if (type == "application/vnd.ollama.image.model" && m.model.empty()) {
            m.model = blob;
        } else if (type == "application/vnd.ollama.image.projector" && m.projector.empty()) {
            m.projector = blob;
        } else if (type == "application/vnd.ollama.image.template" && m.template_file.empty()) {
            m.template_file = blob;
        } else if (type == "application/vnd.ollama.image.system" && m.system_file.empty()) {
            m.system_file = blob;
        } else if (type == "application/vnd.ollama.image.params" && m.params_file.empty()) {
            m.params_file = blob;
        } else if (type == "application/vnd.ollama.image.messages" && m.messages_file.empty()) {
            m.messages_file = blob;
        } else if (type == "application/vnd.ollama.image.adapter") {
            m.adapters.push_back(blob);
        }
    }
}

} // namespace

bool is_projector(const std::string & file) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(file.c_str(), params);
    if (ctx == nullptr) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx, "general.architecture");
    const bool clip = id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING &&
                      std::string(gguf_get_val_str(ctx, id)) == "clip";
    gguf_free(ctx);
    return clip;
}

std::vector<ollama_model> ollama_models(const std::string & folder) {
    std::vector<ollama_model> out;
    const fs::path models = ollama_folder(folder);
    const fs::path manifests = models / "manifests";
    std::error_code ec;
    if (models.empty() || !fs::is_directory(manifests, ec)) {
        return out;
    }
    // manifests/<registry>/<namespace>/<model>/<tag>
    fs::recursive_directory_iterator it(manifests, fs::directory_options::skip_permission_denied, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) {
            continue;
        }
        ollama_model m;
        read_manifest(it->path(), models, m);
        if (m.model.empty() || !fs::is_regular_file(fs::u8path(m.model), ec)) {
            continue;
        }
        if (!m.projector.empty() && !fs::is_regular_file(fs::u8path(m.projector), ec)) {
            m.projector.clear();
        }
        const fs::path rel = fs::relative(it->path(), manifests, ec);
        std::vector<std::string> parts;
        for (const auto & p : rel) {
            parts.push_back(p.u8string());
        }
        if (parts.size() < 4) {
            continue;
        }
        const std::string tag = parts.back();
        const std::string model = parts[parts.size() - 2];
        // the official library shows as "model:tag"; anything else with its registry and namespace
        if (parts[0] == "registry.ollama.ai" && parts[1] == "library") {
            m.name = model + ":" + tag;
        } else {
            std::string n = parts[0];
            for (size_t i = 1; i + 1 < parts.size(); ++i) {
                n += "/" + parts[i];
            }
            m.name = n + ":" + tag;
        }
        out.push_back(m);
    }
    std::sort(out.begin(), out.end(), [](const ollama_model & a, const ollama_model & b) { return a.name < b.name; });
    return out;
}

namespace {

// a GGUF's string value ("" if absent or not a string)
std::string gguf_str(const gguf_context * ctx, const std::string & key) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    return id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING ? gguf_get_val_str(ctx, id) : "";
}

// a name lower-cased with its separators removed: "SmolVLM-256M-Instruct" and "SmolVLM 256M Instruct" compare equal
std::string name_key(std::string s) {
    s = lower(s);
    s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '-' || c == '_' || c == '.' || c == ' '; }),
            s.end());
    return s;
}

// Whether a folder's only reader is its only model's: today's converter writes the reader's general.type and the name
// of the model it was made for (general.name), which must be the model's own name or one of its base models' (a
// fine-tune keeps its base's reader). A reader without them (older converters) is taken only when its file name names
// no model. Its own base_model entries are never used: they name the wrong models (SmolVLM2's reader names
// SmolVLM-256M; SmolVLM-256M's names the text-only SmolLM2-135M).
bool lone_reader_fits(const std::string & reader, const std::string & model) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * r = gguf_init_from_file(reader.c_str(), params);
    if (r == nullptr) {
        return false;
    }
    const std::string r_type = gguf_str(r, "general.type");
    const std::string r_name = name_key(gguf_str(r, "general.name"));
    gguf_free(r);
    if (r_type.empty() || r_name.empty()) {
        return stem_key(reader).size() < 3; // "mmproj-F16.gguf", "mmproj-model-f16.gguf": names no other model
    }
    gguf_context * m = gguf_init_from_file(model.c_str(), params);
    if (m == nullptr) {
        return false;
    }
    bool fits = name_key(gguf_str(m, "general.name")) == r_name;
    const int64_t n_base = gguf_find_key(m, "general.base_model.count");
    const int64_t count = n_base >= 0 && gguf_get_kv_type(m, n_base) == GGUF_TYPE_UINT32 ? gguf_get_val_u32(m, n_base) : 0;
    for (int64_t i = 0; i < count && !fits; ++i) {
        fits = name_key(gguf_str(m, "general.base_model." + std::to_string(i) + ".name")) == r_name;
    }
    gguf_free(m);
    return fits;
}

} // namespace

bool ollama_model_of(const std::string & model, ollama_model & out) {
    std::error_code ec;
    const fs::path path = fs::u8path(model);
    const fs::path dir = path.parent_path();
    if (dir.filename() != "blobs") {
        return false;
    }
    const std::string target = fs::weakly_canonical(path, ec).u8string();
    for (const auto & m : ollama_models(dir.parent_path().u8string())) {
        if (fs::weakly_canonical(fs::u8path(m.model), ec).u8string() == target) {
            out = m;
            return true;
        }
    }
    return false;
}

bool has_chat_template(const std::string & gguf) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * ctx = gguf_init_from_file(gguf.c_str(), params);
    if (ctx == nullptr) {
        return false;
    }
    const int64_t id = gguf_find_key(ctx, "tokenizer.chat_template");
    const bool has = id >= 0 && gguf_get_kv_type(ctx, id) == GGUF_TYPE_STRING && gguf_get_val_str(ctx, id)[0] != '\0';
    gguf_free(ctx);
    return has;
}

std::string find_projector(const std::string & model, std::string * left_out) {
    std::error_code ec;
    const fs::path path = fs::u8path(model);
    const fs::path dir = path.parent_path();
    // a blob in Ollama's store: what its manifest lists with it
    if (dir.filename() == "blobs") {
        const std::string target = fs::weakly_canonical(path, ec).u8string();
        for (const auto & m : ollama_models(dir.parent_path().u8string())) {
            if (!m.projector.empty() && fs::weakly_canonical(fs::u8path(m.model), ec).u8string() == target &&
                is_projector(m.projector)) {
                return m.projector;
            }
        }
        return "";
    }
    // next to the model: the image readers there, the closest name first
    std::vector<std::string> found;
    for (fs::directory_iterator it(dir.empty() ? fs::path(".") : dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = lower(it->path().filename().u8string());
        if (it->is_regular_file(ec) && name.find("mmproj") != std::string::npos && name.size() > 5 &&
            name.substr(name.size() - 5) == ".gguf" && it->path() != path && is_projector(it->path().u8string())) {
            found.push_back(it->path().u8string());
        }
    }
    if (found.empty()) {
        return "";
    }
    const std::string key = stem_key(model);
    std::sort(found.begin(), found.end());
    // only a reader made for this model (names_match), the one whose name is closest first; or the folder's only
    // reader beside the folder's only model (a download's own folder, where readers are often named just
    // "mmproj-F16.gguf"). Another model's reader makes llama.cpp stop, or worse, load and read pictures wrongly.
    std::vector<std::string> mine;
    for (const auto & f : found) {
        if (names_match(stem_key(f), key)) {
            mine.push_back(f);
        }
    }
    if (!mine.empty()) {
        // the closest name first; among one model's readers in several formats, the one whose precision is nearest the
        // model file's, then the more precise (llama.cpp's own rule for its downloads)
        const int bits = quant_bits(model);
        auto far = [&](const std::string & f) {
            const int b = quant_bits(f);
            return bits < 0 || b < 0 ? 0 : std::abs(b - bits);
        };
        std::stable_sort(mine.begin(), mine.end(), [&](const std::string & a, const std::string & b) {
            const size_t pa = common_prefix(stem_key(a), key), pb = common_prefix(stem_key(b), key);
            if (pa != pb) {
                return pa > pb;
            }
            if (far(a) != far(b)) {
                return far(a) < far(b);
            }
            return quant_bits(a) > quant_bits(b);
        });
        return mine.front();
    }
    if (found.size() == 1) {
        size_t models = 0;
        for (fs::directory_iterator it(dir.empty() ? fs::path(".") : dir, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string name = lower(it->path().filename().u8string());
            const bool later_part = name.find("-of-") != std::string::npos && name.find("-00001-of-") == std::string::npos;
            if (it->is_regular_file(ec) && name.size() > 5 && name.substr(name.size() - 5) == ".gguf" &&
                name.find("mmproj") == std::string::npos && !later_part) {
                ++models;
            }
        }
        if (models == 1) {
            if (lone_reader_fits(found.front(), model)) {
                return found.front();
            }
            if (left_out != nullptr) {
                *left_out = found.front();
            }
        }
    }
    return "";
}

} // namespace kera
