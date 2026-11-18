// Model cards' recommended settings (kr_model_defaults.h).

#include "tune/kr_model_defaults.h"

#include <algorithm>
#include <cctype>

namespace kera {

namespace {

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool has(const std::string & s, const char * part) {
    return s.find(part) != std::string::npos;
}

} // namespace

const char * family_name(model_family f) {
    switch (f) {
        case model_family::nemotron3:   return "Nemotron 3 Nano";
        case model_family::qwen3:       return "Qwen3";
        case model_family::qwen3_coder: return "Qwen3-Coder";
        default:                        return "unknown";
    }
}

model_family family_of(const std::string & arch, const std::string & name) {
    const std::string n = lower(name);
    // Nemotron 3 Nano 30B-A3B, the MoE hybrid whose card was read. Not the rest of the family: Super, Ultra and the 3.5
    // releases recommend 1.0/0.95 for every request, Nano Omni 0.6/0.95 thinking and 0.2/top_k 1 otherwise; the dense
    // nemotron_h models (Nemotron-H, Nemotron Nano 2) have cards of their own. Their files carry their own values.
    if (arch == "nemotron_h_moe" && !has(n, "omni") &&
        (has(n, "nemotron 3 nano") || has(n, "nemotron-3-nano") || has(n, "nemotron_3_nano") ||
         (has(n, "nemotron nano 3") && !has(n, "nemotron nano 3.")))) {
        return model_family::nemotron3;
    }
    // Qwen's own releases are named "Qwen3 ..." / "Qwen3-..."; a model built on Qwen3 under another name (such as
    // DeepSeek-R1-0528-Qwen3-8B) has its own card
    if ((arch == "qwen3" || arch == "qwen3moe") && n.rfind("qwen3", 0) == 0) {
        return has(n, "coder") ? model_family::qwen3_coder : model_family::qwen3;
    }
    return model_family::unknown;
}

bool recommended_sampling(model_family f, const std::string & name, bool thinking, bool tools, sampling_values & out) {
    out = sampling_values{};
    const std::string n = lower(name);
    switch (f) {
        case model_family::nemotron3:
            // NVIDIA's card: temperature 1.0 and top_p 1.0 with reasoning on, greedy with it off,
            // temperature 0.6 and top_p 0.95 for tool calling. It names no top_k or min_p, which NVIDIA's own serving
            // engines leave off: so off here too, not llama.cpp's 40 and 0.05
            if (tools) {
                out.temp = 0.6f;
                out.top_p = 0.95f;
            } else if (thinking) {
                out.temp = 1.0f;
                out.top_p = 1.0f;
            } else {
                out.temp = 0.0f;
            }
            out.top_k = 0;
            out.min_p = 0.0f;
            out.top_k_from_engine = out.min_p_from_engine = true;
            return true;
        case model_family::qwen3:
            // Qwen's card: thinking 0.6 / 0.95 / 20 / 0; non-thinking 0.7 / 0.8 / 20 / 0. The 2507
            // Thinking release always thinks, even when its template has no switch
            if (thinking || has(n, "thinking")) {
                out.temp = 0.6f;
                out.top_p = 0.95f;
            } else {
                out.temp = 0.7f;
                out.top_p = 0.8f;
            }
            out.top_k = 20;
            out.min_p = 0.0f;
            return true;
        case model_family::qwen3_coder:
            // Qwen3-Coder's card: temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.05. It names no min_p,
            // which Qwen's own engines leave off: so off here too, not llama.cpp's 0.05
            out.temp = 0.7f;
            out.top_p = 0.8f;
            out.top_k = 20;
            out.repeat = 1.05f;
            out.min_p = 0.0f;
            out.min_p_from_engine = true;
            return true;
        default:
            return false;
    }
}

float recommended_presence_penalty(model_family f, const std::string & name, bool quantized) {
    return f == model_family::qwen3 && quantized && !has(lower(name), "2507") ? 1.5f : -1.0f;
}

bool card_agrees_with_file(model_family f, const std::string & name, const sampling_values & file) {
    const auto same = [](float a, float b) { return a - b < 1e-3f && b - a < 1e-3f; };
    // the card's plain settings, thinking or not (a file's generation_config.json describes plain generation, not tool
    // calls: a 0.6/0.95 file under Nemotron 3 Nano's name is another model's, even though Nano uses 0.6/0.95 for tools)
    for (const bool thinking : { true, false }) {
        for (const bool tools : { false }) {
            sampling_values c;
            if (!recommended_sampling(f, name, thinking, tools, c)) {
                continue;
            }
            // a value the file sets must be the card's, where the card gives one
            // (values filled from the makers' engines aren't the card's own: not compared)
            const bool ok = (file.temp < 0 || c.temp < 0 || same(file.temp, c.temp)) &&
                            (file.top_p < 0 || c.top_p < 0 || same(file.top_p, c.top_p)) &&
                            (file.top_k < 0 || c.top_k < 0 || c.top_k_from_engine || file.top_k == c.top_k) &&
                            (file.min_p < 0 || c.min_p < 0 || c.min_p_from_engine || same(file.min_p, c.min_p));
            if (ok) {
                return true;
            }
        }
    }
    return false;
}

namespace {

struct listed_card {
    const char * key;  // the repository's name in lower case, letters and digits only
    int layers, width;
    float temp, top_p;
    int top_k;
    float min_p, repeat; // < 0: the makers name none
    const char * repo;
};

const listed_card listed_cards[] = {
#include "tune/model_cards.inc"
};

std::string letters_and_digits(const std::string & s) {
    std::string out;
    for (const char c : s) {
        if (std::isalnum((unsigned char) c)) {
            out += (char) std::tolower((unsigned char) c);
        }
    }
    return out;
}

} // namespace

bool listed_sampling(const std::vector<std::string> & names, int n_layer, int n_embd, sampling_values & out,
                     std::string & repo) {
    out = sampling_values{};
    if (n_layer <= 0 || n_embd <= 0) {
        return false;
    }
    for (const auto & name : names) {
        const std::string key = letters_and_digits(name);
        if (key.size() < 4) {
            continue;
        }
        for (const auto & c : listed_cards) {
            if (c.layers == n_layer && c.width == n_embd && key == c.key) {
                out.temp = c.temp;
                out.top_p = c.top_p;
                out.top_k = c.top_k;
                out.min_p = c.min_p;
                out.repeat = c.repeat;
                repo = c.repo;
                return true;
            }
        }
    }
    return false;
}

} // namespace kera
