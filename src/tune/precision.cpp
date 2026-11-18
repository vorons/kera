// A model file's precision from its name (kr_precision.h).

#include "tune/kr_precision.h"

#include <cctype>

namespace kera {

namespace {

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

bool delim(char c) {
    return c == '-' || c == '_' || c == '.' || c == ' ';
}

// the rank of a precision tag starting at s[at] and its length; -1 if none starts there
int tag_at(const std::string & s, size_t at, size_t & len) {
    const auto starts = [&](const char * w) { return s.compare(at, std::char_traits<char>::length(w), w) == 0; };
    const auto word_end = [&](size_t from) { // letters, digits and '_' belong to the tag ("q4_k_m", "fp8_e4m3fn")
        size_t e = from;
        while (e < s.size() && (std::isalnum((unsigned char) s[e]) || s[e] == '_')) {
            ++e;
        }
        return e;
    };
    for (const char * w : { "f32", "fp32" }) {
        if (starts(w) && (at + std::char_traits<char>::length(w) == s.size() ||
                          delim(s[at + std::char_traits<char>::length(w)]))) {
            len = std::char_traits<char>::length(w);
            return 0;
        }
    }
    for (const char * w : { "bf16", "fp16", "f16" }) {
        if (starts(w) && (at + std::char_traits<char>::length(w) == s.size() ||
                          delim(s[at + std::char_traits<char>::length(w)]))) {
            len = std::char_traits<char>::length(w);
            return 1;
        }
    }
    if (starts("fp8")) {
        len = word_end(at) - at;
        return 2;
    }
    // "q8_0", "q4_k_m", "iq4_xs", and unsloth's "ud-q4_k_xl"
    size_t q = at;
    if (starts("ud-") || starts("ud_")) {
        q += 3;
    }
    if (q < s.size() && s[q] == 'i' && q + 1 < s.size() && s[q + 1] == 'q') {
        ++q;
    }
    if (q + 1 < s.size() && s[q] == 'q' && std::isdigit((unsigned char) s[q + 1])) {
        len = word_end(q) - at;
        const int bits = s[q + 1] - '0';
        return bits >= 8 ? 2 : bits == 6 ? 3 : bits == 5 ? 4 : bits == 4 ? 5 : bits == 3 ? 6 : 7;
    }
    return -1;
}

std::string extension(const std::string & n) {
    const size_t dot = n.rfind('.');
    return dot == std::string::npos ? "" : n.substr(dot);
}

} // namespace

int precision_of(const std::string & file_name, std::string * stem) {
    std::string n = lower(file_name);
    const std::string ext = extension(n);
    if (ext == ".gguf" || ext == ".safetensors" || ext == ".ckpt" || ext == ".sft") {
        n.resize(n.size() - ext.size());
    }
    int rank = -1;
    for (size_t at = 0; at < n.size(); ++at) {
        if (at > 0 && !delim(n[at - 1])) {
            continue;
        }
        size_t len = 0;
        const int r = tag_at(n, at, len);
        if (r < 0) {
            continue;
        }
        rank = r;
        // the tag and one delimiter before it (or after it, at the start of the name) come out of the stem
        size_t from = at, to = at + len;
        if (from > 0) {
            --from;
        } else if (to < n.size() && delim(n[to])) {
            ++to;
        }
        n.erase(from, to - from);
        break;
    }
    if (stem != nullptr) {
        *stem = n;
    }
    return rank;
}

std::string higher_precision_sibling(const std::string & name, const std::vector<sibling_file> & folder,
                                     uint64_t fit_bytes) {
    std::string stem;
    const int mine = precision_of(name, &stem);
    if (mine <= 1) {
        return ""; // already the model's own weights, or no tag to tell by
    }
    const std::string ext = extension(lower(name));
    std::string best;
    int best_rank = mine;
    for (const auto & f : folder) {
        std::string other;
        const int r = precision_of(f.name, &other);
        // 16-bit is the model's own weights for nearly every model: a 32-bit copy of it adds nothing
        const int rank = r == 0 ? 1 : r;
        if (r < 0 || other != stem || extension(lower(f.name)) != ext || rank >= best_rank ||
            (fit_bytes > 0 && f.bytes > fit_bytes)) {
            continue;
        }
        best = f.name;
        best_rank = rank;
    }
    return best;
}

} // namespace kera
