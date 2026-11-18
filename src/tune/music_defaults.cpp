// The music folder's models, chosen as ACE-Step's own guide recommends (kr_music_defaults.h).

#include "tune/kr_music_defaults.h"

#include <algorithm>
#include <cctype>
#include <cstdio>

namespace kera {

namespace {

std::string lower(std::string s) {
    for (auto & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

// A name's part between delimiters, as ACE-Step's own interface tells its models apart (model_config.py _has_token)
bool has_token(const std::string & n, const char * token) {
    const std::string t = token;
    for (size_t at = n.find(t); at != std::string::npos; at = n.find(t, at + 1)) {
        const auto delim = [](char c) { return c == '-' || c == '_' || c == '.' || c == '/' || c == '\\'; };
        const bool left = at == 0 || delim(n[at - 1]);
        const bool right = at + t.size() == n.size() || delim(n[at + t.size()]);
        if (left && right) {
            return true;
        }
    }
    return false;
}

// 0 (highest) to 9: how much of the weights' precision a file keeps, from its name
int precision_rank(const std::string & n) {
    static const char * const order[] = { "f32", "bf16", "f16", "q8_0", "q6_k", "q5_k_m", "q5_k_s", "q5_0",
                                          "q4_k_m", "q4_k_s", "q4_0", "q3_k_l", "q3_k_m", "q3_k_s", "q2_k" };
    int rank = 0;
    for (const char * q : order) {
        if (n.find(q) != std::string::npos) {
            return rank;
        }
        ++rank;
    }
    return rank; // unnamed: after the named ones
}

// The best file of those with the best (lowest) preference: the highest precision that leaves the memory room, else
// the smallest. prefs[i] < 0 leaves file i out.
int best_of(const std::vector<music_file> & files, const std::vector<int> & prefs, double memory_gb) {
    int best_pref = -1;
    for (const int p : prefs) {
        if (p >= 0 && (best_pref < 0 || p < best_pref)) {
            best_pref = p;
        }
    }
    if (best_pref < 0) {
        return -1;
    }
    int pick = -1, smallest = -1;
    for (size_t i = 0; i < files.size(); ++i) {
        if (prefs[i] != best_pref) {
            continue;
        }
        if (smallest < 0 || files[i].bytes < files[(size_t) smallest].bytes) {
            smallest = (int) i;
        }
        // the engine holds one model at a time; half the memory is left for its working memory and the rest
        const bool fits = memory_gb <= 0 || (double) files[i].bytes / 1e9 <= 0.5 * memory_gb;
        if (fits) {
            const int mine = precision_rank(lower(files[i].name));
            const int best = pick < 0 ? 1000 : precision_rank(lower(files[(size_t) pick].name));
            // at the same precision, the plain release ("turbo") before its variants ("turbo-shift1", "turbo-continuous")
            if (mine < best || (mine == best && files[i].name.size() < files[(size_t) pick].name.size())) {
                pick = (int) i;
            }
        }
    }
    return pick >= 0 ? pick : smallest;
}

} // namespace

music_choice choose_music_models(const std::vector<music_file> & lms, const std::vector<music_file> & dits,
                                 double gpu_gb, double ram_gb) {
    music_choice out;
    // ACE-Step's guide is written for graphics memory. On the CPU the planner follows the computer's memory (it runs
    // once per song), and the sound model the guide's smallest row (turbo: 8 model calls a song instead of 100).
    const double lm_gb = gpu_gb > 0 ? gpu_gb : ram_gb;
    const double dit_gb = gpu_gb;

    // the planner: 4B from 24 GB, 1.7B from 12 GB, 0.6B below; then the nearest smaller one, then the nearest larger
    std::vector<int> lm_pref(lms.size(), -1);
    for (size_t i = 0; i < lms.size(); ++i) {
        const std::string n = lower(lms[i].name);
        const int size = n.find("0.6b") != std::string::npos ? 0 : n.find("1.7b") != std::string::npos ? 1 :
                         n.find("4b") != std::string::npos ? 2 : -1;
        if (n.rfind("acestep-5hz-lm", 0) != 0 || size < 0) {
            lm_pref[i] = 9; // not one of ACE-Step's own names: only when nothing else is there
            continue;
        }
        const int want = lm_gb >= 24 ? 2 : lm_gb >= 12 ? 1 : 0;
        lm_pref[i] = size == want ? 0 : size < want ? want - size : 3 + size - want;
    }
    // the sound model, by the guide's rows: from 24 GB XL SFT; 20 to 24 GB XL turbo or XL SFT; 16 to 20 GB SFT or XL
    // turbo; below, turbo or SFT. The base models last (the guide names them for extract, lego and complete only).
    std::vector<int> dit_pref(dits.size(), -1);
    for (size_t i = 0; i < dits.size(); ++i) {
        const std::string n = lower(dits[i].name);
        if (n.rfind("acestep-v15", 0) != 0) {
            dit_pref[i] = 9;
            continue;
        }
        const bool xl = has_token(n, "xl");
        const bool turbo = has_token(n, "turbo");
        const bool sft = has_token(n, "sft") && !turbo;
        const int kind = turbo ? (xl ? 1 : 0) : sft ? (xl ? 3 : 2) : (xl ? 5 : 4); // turbo, xl-turbo, sft, xl-sft, base, xl-base
        static const int rows[4][6] = {
            // turbo xl-turbo sft xl-sft base xl-base
            { 0, 2, 1, 3, 4, 5 }, // below 16 GB, and on the CPU
            { 2, 1, 0, 3, 4, 5 }, // 16 to 20 GB
            { 2, 0, 3, 1, 5, 4 }, // 20 to 24 GB
            { 3, 1, 2, 0, 5, 4 }, // from 24 GB
        };
        dit_pref[i] = rows[dit_gb >= 24 ? 3 : dit_gb >= 20 ? 2 : dit_gb >= 16 ? 1 : 0][kind];
    }
    const int lm = best_of(lms, lm_pref, lm_gb);
    const int dit = best_of(dits, dit_pref, gpu_gb > 0 ? gpu_gb : ram_gb);
    if (lm >= 0) {
        out.lm = lms[(size_t) lm].name;
    }
    if (dit >= 0) {
        out.dit = dits[(size_t) dit].name;
    }
    if (lms.size() > 1 || dits.size() > 1) {
        char memory[64];
        if (gpu_gb > 0) {
            std::snprintf(memory, sizeof(memory), "%.0f GB of graphics memory", gpu_gb);
        } else {
            std::snprintf(memory, sizeof(memory), "the CPU and %.0f GB of memory", ram_gb);
        }
        out.why = std::string("of the folder's models, the ones ACE-Step's guide recommends for ") + memory;
    }
    return out;
}

} // namespace kera
