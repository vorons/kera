// `kera serve`: llama.cpp's server (OpenAI and Anthropic APIs, tool calling, thinking switch), plus
// Kera's saved chat states on the SSD (spec 4b item 5): the server's RAM prompt cache gets a second
// tier, so going back to an old chat or restarting doesn't mean reading the whole chat again.
//
// Saved states only fit the exact model file and KV settings they came from, so they live in a folder
// named after the model file's hash and those settings (the cache identity, 4b item 5).
//
// A MoE model bigger than the memory the machine profile allows has its experts streamed from the SSD
// (M5, src/stream): --stream off|os|kera overrides that, --stream-budget-gb sets Kera's cache.
//
// Several chats at once (D-023, llama.cpp hook 7; --multi-user, or the serve.* settings): a new message's prompt is
// read before the rest of a long one, reading a long prompt takes at most a share of the chip's time while others
// write, and a start a chat has in common with another chat's memory is shared rather than read again.
//
// Kera never changes how a model answers; it only makes it faster (D-033). It has no modes that trade quality for
// speed: the opt-in lossy modes were removed (D-034).

#include "app/kera_app.h"
#include "common/kr_hash.h"
#include "common/kr_paths.h"
#include "kvstore/kr_prompt_disk.h"
#include "planner/chat_share.h"
#include "planner/kr_plan.h"
#include "planner/projector.h"
#include "setup/kr_machine.h"
#include "stream/kr_stream.h"
#include "tune/kr_settings.h"
#include "kera_version.h"

#include "batch-scheduler.h"
#include "common.h"
#include "prompt-store.h"
#include "request-defaults.h"
#include "spec-controller.h"
#include "tune/kr_model_defaults.h"
#include "tune/kr_precision.h"
#include "spec/kr_controller.h"
#include "spec/kr_guess_memory.h"
#include "spec/kr_guess_switch.h"
#include "speculative.h"

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <csignal>
#include <unistd.h>
#endif

int llama_server(int argc, char ** argv);
int llama_cli(int argc, char ** argv);

namespace kera {

namespace {

// Converts between the server's saved prompts and the disk store's opaque blobs:
//   meta  = [n_checkpoints, then (n_tokens, pos_min, pos_max) per checkpoint]
//   blobs = [state_tgt, state_dft, then (data_tgt, data_dft, data_spec) per checkpoint]
class server_store_adapter : public common_prompt_store {
public:
    // The store's folder needs the model's hash: until it is known (hashing a big model takes a little
    // while the first time), saves wait in memory and loads find nothing.
    void open(const std::string & dir, uint64_t budget) {
        auto disk = std::make_shared<prompt_disk_store>(dir, budget);
        std::vector<saved_prompt_blobs> waiting;
        {
            std::lock_guard<std::mutex> lock(mu_);
            disk_ = disk;
            waiting.swap(waiting_);
        }
        for (auto & w : waiting) {
            disk->save_async(std::move(w));
        }
    }

    bool has_waiting() {
        std::lock_guard<std::mutex> lock(mu_);
        return !waiting_.empty();
    }

    // the server, once it has loaded the model: what its saved states depend on besides the model file
    void set_identity(const std::string & settings, const std::vector<std::string> & files) override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (identity_set_) {
                return; // loaded again after sleeping: the same settings
            }
            settings_ = settings;
            files_ = files;
            identity_set_ = true;
        }
        identity_cv_.notify_all();
    }

    // waits for it; false if the server stops first
    bool wait_identity(const std::atomic<bool> & running, std::string & settings, std::vector<std::string> & files) {
        std::unique_lock<std::mutex> lock(mu_);
        while (!identity_set_) {
            if (!running) {
                return false;
            }
            identity_cv_.wait_for(lock, std::chrono::milliseconds(200));
        }
        settings = settings_;
        files = files_;
        return true;
    }

    void save(common_saved_prompt && p) override {
        saved_prompt_blobs b;
        b.tokens.assign(p.tokens.begin(), p.tokens.end());
        b.meta.push_back((int64_t) p.checkpoints.size());
        b.blobs.push_back({ p.state_tgt.begin(), p.state_tgt.end() });
        b.blobs.push_back({ p.state_dft.begin(), p.state_dft.end() });
        for (const auto & cp : p.checkpoints) {
            auto c = *cp;
            b.meta.push_back(c.n_tokens);
            b.meta.push_back(c.pos_min);
            b.meta.push_back(c.pos_max);
            b.blobs.push_back({ c.data_tgt.begin(), c.data_tgt.end() });
            b.blobs.push_back({ c.data_dft.begin(), c.data_dft.end() });
            b.blobs.push_back(std::move(c.data_spec));
        }
        std::shared_ptr<prompt_disk_store> disk;
        {
            std::lock_guard<std::mutex> lock(mu_);
            disk = disk_;
            if (!disk) {
                waiting_.clear(); // only the newest state is worth keeping in memory
                waiting_.push_back(std::move(b));
                return;
            }
        }
        disk->save_async(std::move(b));
    }

    bool load(const std::vector<llama_token> & tokens_new, size_t min_lcp, common_saved_prompt & out) override {
        std::shared_ptr<prompt_disk_store> disk;
        {
            std::lock_guard<std::mutex> lock(mu_);
            disk = disk_;
        }
        if (!disk) {
            return false;
        }
        const std::vector<int32_t> t(tokens_new.begin(), tokens_new.end());
        saved_prompt_blobs b;
        size_t lcp = 0;
        if (!disk->load_best(t, min_lcp, 0.25, b, &lcp)) {
            return false;
        }
        const int64_t n = b.meta.empty() ? -1 : b.meta[0];
        if (n < 0 || (int64_t) b.meta.size() != 1 + 3 * n || (int64_t) b.blobs.size() != 2 + 3 * n) {
            return false;
        }
        out.tokens.assign(b.tokens.begin(), b.tokens.end());
        out.state_tgt = common_state_data(b.blobs[0].begin(), b.blobs[0].end());
        out.state_dft = common_state_data(b.blobs[1].begin(), b.blobs[1].end());
        out.checkpoints.clear();
        for (int64_t i = 0; i < n; ++i) {
            common_prompt_checkpoint c;
            c.n_tokens = b.meta[(size_t) (1 + 3 * i)];
            c.pos_min = (llama_pos) b.meta[(size_t) (2 + 3 * i)];
            c.pos_max = (llama_pos) b.meta[(size_t) (3 + 3 * i)];
            c.id_task = -1; // restored from disk: the server falls back to re-reading if it can't load it
            c.data_tgt = common_state_data(b.blobs[(size_t) (2 + 3 * i)].begin(), b.blobs[(size_t) (2 + 3 * i)].end());
            c.data_dft = common_state_data(b.blobs[(size_t) (3 + 3 * i)].begin(), b.blobs[(size_t) (3 + 3 * i)].end());
            c.data_spec = std::move(b.blobs[(size_t) (4 + 3 * i)]);
            out.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(std::move(c)));
        }
        std::fprintf(stderr, "Kera: reused %zu tokens of a saved chat from the SSD\n", lcp);
        return true;
    }

private:
    std::mutex mu_;
    std::shared_ptr<prompt_disk_store> disk_;
    std::vector<saved_prompt_blobs> waiting_;
    std::condition_variable identity_cv_;
    bool identity_set_ = false;
    std::string settings_;
    std::vector<std::string> files_;
};

// Guess-ahead (spec 11.1, Appendix B) and the guesser switch (11.1 item 1): for each chat, picks at every
// step which of llama.cpp's guessers guesses and how far, by measuring how many words each choice really
// produces per second, "don't guess" included, so guessing never makes writing slower than without it
// (src/spec/kr_guess_switch.h).
class guess_controller : public common_spec_controller {
public:
    // KERA_GUESS_FIXED_K pins every guesser's length (for measuring; 0 = never guess, with the guessers
    // still loaded)
    guess_controller() {
        if (const char * k = std::getenv("KERA_GUESS_FIXED_K"); k != nullptr && *k != '\0') {
            fixed_k_ = std::max(0, std::atoi(k));
        }
    }

    int32_t draft_len(llama_seq_id seq, int32_t n_max) override {
        std::lock_guard<std::mutex> lock(mu_);
        const bool try_now = get(seq).step(now_s());
        return try_now || fixed_k_ > 0 ? n_max : 0; // each guesser's length is chosen in guesser_len
    }

    int32_t guesser_len(llama_seq_id seq, int32_t type, bool drafted, int32_t n_max) override {
        std::lock_guard<std::mutex> lock(mu_);
        const int32_t k = get(seq).ask(type, drafted, n_max);
        return fixed_k_ >= 0 ? std::min(n_max, (int32_t) fixed_k_) : k;
    }

    void accepted(llama_seq_id seq, int32_t n_drafted, int32_t n_accepted) override {
        (void) n_drafted;
        std::lock_guard<std::mutex> lock(mu_);
        get(seq).accepted(n_accepted);
    }

    // Options for chats that start after this (e.g. lower starts when the experts are streamed)
    void set_options(const guess_switch::options & o) {
        std::lock_guard<std::mutex> lock(mu_);
        options_ = o;
    }

    // Remember, per model, model guessers that don't pay (src/spec/kr_guess_memory.h)
    void remember(guess_memory * memory, std::string model_key) {
        std::lock_guard<std::mutex> lock(mu_);
        memory_ = memory;
        model_key_ = std::move(model_key);
    }

    void reset(llama_seq_id seq) override {
        std::lock_guard<std::mutex> lock(mu_);
        auto & sw = get(seq);
        sw.stop(); // keep what it learned: it holds for the next reply
        std::string line;
        for (const auto & g : sw.stats()) {
            // a guesser that runs a model costs time while loaded: note whether it pays, once it's clear
            if (memory_ != nullptr && !g.free && g.steps >= 1000 && !g.testing) {
                guess_verdict v;
                v.pays = g.k > 0;
                v.time = (int64_t) std::chrono::duration_cast<std::chrono::seconds>(
                             std::chrono::system_clock::now().time_since_epoch()).count();
                v.steps = g.steps;
                v.words_per_s = g.seconds > 0 ? g.tokens / g.seconds : 0.0;
                memory_->set(model_key_, common_speculative_type_to_str((common_speculative_type) g.type), v);
            }
            if (g.steps > 0) {
                char buf[160];
                std::snprintf(buf, sizeof(buf), "%s%s: %lld steps, %.1f words/s, now guessing %d", line.empty() ? "" : "; ",
                              guesser_name(g.type), g.steps, g.seconds > 0 ? g.tokens / g.seconds : 0.0, g.k);
                line += buf;
            }
        }
        if (!line.empty()) {
            std::fprintf(stderr, "Kera: guessing so far in this chat: %s\n", line.c_str());
        }
    }

private:
    static double now_s() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static const char * guesser_name(int type) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft model";
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "EAGLE-3";
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "MTP head";
            case COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH:  return "DFlash";
            case COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK:  return "DSpark";
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "copying (n-gram simple)";
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "copying (n-gram map)";
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "copying (n-gram map k4v)";
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "copying (n-gram mod)";
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "copying (n-gram cache)";
            default:                                    return "guesser";
        }
    }

    guess_switch & get(llama_seq_id seq) {
        auto & p = seqs_[seq];
        if (!p) {
            p = std::make_unique<guess_switch>(options_);
        }
        return *p;
    }

    std::mutex mu_;
    std::map<llama_seq_id, std::unique_ptr<guess_switch>> seqs_;
    int fixed_k_ = -1;
    guess_switch::options options_;
    guess_memory * memory_ = nullptr;
    std::string model_key_;
};

// Several chats sharing the chip (hook 7, src/planner/chat_share.h): the order prompts are read in, how much prompt
// reading a step takes while others write, and whether a chat may share another's start
class share_scheduler : public common_batch_scheduler {
public:
    share_scheduler(const chat_share_options & o, int32_t share_min) : policy_(o), share_min_(share_min) {}

    int32_t plan(std::vector<prompt> & prompts, int32_t n_writing, int32_t n_batch) override {
        std::vector<chat_share::waiting> w;
        w.reserve(prompts.size());
        for (const auto & p : prompts) {
            w.push_back({ p.slot, p.left, p.since, p.priority });
        }
        const int32_t budget = policy_.plan(w, n_writing, n_batch);
        for (size_t i = 0; i < w.size(); ++i) {
            prompts[i] = { w[i].slot, w[i].left, w[i].since, w[i].priority };
        }
        return budget;
    }

    void step(int32_t n_prompt, int32_t n_writing, double ms) override {
        policy_.step(n_prompt, n_writing, ms / 1000);
    }

    int32_t share_min_tokens() const override {
        return share_min_;
    }

private:
    chat_share policy_;
    int32_t share_min_;
};

// What a model's packaging gives every chat besides its weights: an Ollama Modelfile's PARAMETERs (as Ollama applies
// them to every request), its SYSTEM text and MESSAGE pairs (rule one, D-033: a model from Ollama's store runs the way
// Ollama runs it)
struct packaging_defaults {
    common_sampling_defaults sampling; // < 0 / empty where the Modelfile sets none
    bool any_sampling = false;
    std::string system;
    std::vector<std::pair<std::string, std::string>> messages;
};

// The sampling the model's card recommends, per request (hook 4, spec 4c item 2), with the packaging's own settings
// over it where it has them; the request's and the user's own settings win over both (server-common.cpp)
class model_request_defaults : public common_request_defaults {
public:
    model_request_defaults(model_family f, std::string name, packaging_defaults pack, float presence,
                           sampling_values listed = sampling_values{}, bool has_listed = false)
        : family_(f), name_(std::move(name)), pack_(std::move(pack)), presence_(presence), listed_(listed),
          has_listed_(has_listed) {}

    bool sampling(bool thinking, bool tools, common_sampling_defaults & d) override {
        bool any = false;
        sampling_values v;
        if (family_ != model_family::unknown && recommended_sampling(family_, name_, thinking, tools, v)) {
            d.temp = v.temp;
            d.top_p = v.top_p;
            d.top_k = v.top_k;
            d.min_p = v.min_p;
            d.penalty_repeat = v.repeat;
            d.penalty_present = presence_;
            any = true;
        } else if (has_listed_) {
            // the makers' own generation_config.json (D-036): only the values it names, as a file converted from the
            // makers' model carries them; the rest stay the server's
            d.temp = listed_.temp;
            d.top_p = listed_.top_p;
            d.top_k = listed_.top_k;
            d.min_p = listed_.min_p;
            d.penalty_repeat = listed_.repeat;
            any = true;
        }
        const common_sampling_defaults & p = pack_.sampling;
        if (pack_.any_sampling) {
            d.temp = p.temp >= 0 ? p.temp : d.temp;
            d.top_p = p.top_p >= 0 ? p.top_p : d.top_p;
            d.top_k = p.top_k >= 0 ? p.top_k : d.top_k;
            d.min_p = p.min_p >= 0 ? p.min_p : d.min_p;
            d.penalty_repeat = p.penalty_repeat >= 0 ? p.penalty_repeat : d.penalty_repeat;
            d.penalty_present = p.penalty_present >= 0 ? p.penalty_present : d.penalty_present;
            d.penalty_freq = p.penalty_freq >= 0 ? p.penalty_freq : d.penalty_freq;
            d.stop = p.stop;
            any = true;
        }
        return any;
    }

    // Ollama's rule: the Modelfile's SYSTEM first where the chat has no system message, then its MESSAGE pairs
    std::vector<std::pair<std::string, std::string>> leading_messages(bool request_has_system) override {
        std::vector<std::pair<std::string, std::string>> out;
        if (!request_has_system && !pack_.system.empty()) {
            out.emplace_back("system", pack_.system);
        }
        out.insert(out.end(), pack_.messages.begin(), pack_.messages.end());
        return out;
    }

private:
    model_family family_;
    std::string name_;
    packaging_defaults pack_;
    float presence_; // the card's presence penalty for this file, < 0 for none
    sampling_values listed_; // the makers' generation_config.json values, for a model with no card read by hand
    bool has_listed_;
};

// An Ollama Modelfile's settings and messages, from its manifest's layers (only what changes how the model answers:
// num_ctx, num_predict and the like are Ollama's runtime choices and would shorten its memory or its answers)
packaging_defaults ollama_packaging(const ollama_model & m, bool sampling_on) {
    packaging_defaults p;
    auto text = [](const std::string & file) {
        std::ifstream f(std::filesystem::u8path(file), std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    };
    if (!m.system_file.empty()) {
        p.system = text(m.system_file);
    }
    if (!m.messages_file.empty()) {
        const nlohmann::json j = nlohmann::json::parse(text(m.messages_file), nullptr, false);
        for (const auto & msg : j.is_array() ? j : nlohmann::json::array()) {
            if (msg.is_object() && msg.contains("role") && msg.contains("content") && msg["role"].is_string() &&
                msg["content"].is_string()) {
                p.messages.emplace_back(msg["role"].get<std::string>(), msg["content"].get<std::string>());
            }
        }
    }
    if (sampling_on && !m.params_file.empty()) {
        const nlohmann::json j = nlohmann::json::parse(text(m.params_file), nullptr, false);
        auto num = [&](const char * key, float & out) {
            if (j.is_object() && j.contains(key) && j[key].is_number()) {
                out = j[key].get<float>();
                p.any_sampling = true;
            }
        };
        num("temperature", p.sampling.temp);
        num("top_p", p.sampling.top_p);
        num("min_p", p.sampling.min_p);
        num("repeat_penalty", p.sampling.penalty_repeat);
        num("presence_penalty", p.sampling.penalty_present);
        num("frequency_penalty", p.sampling.penalty_freq);
        if (j.is_object() && j.contains("top_k") && j["top_k"].is_number_integer()) {
            p.sampling.top_k = j["top_k"].get<int>();
            p.any_sampling = true;
        }
        if (j.is_object() && j.contains("stop") && j["stop"].is_array()) {
            for (const auto & s : j["stop"]) {
                if (s.is_string()) {
                    p.sampling.stop.push_back(s.get<std::string>());
                    p.any_sampling = true;
                }
            }
        }
    }
    return p;
}

// The value after a flag, e.g. "-m FILE" or "--model FILE"
std::string flag_value(int argc, char ** argv, std::initializer_list<const char *> names) {
    for (int i = 1; i + 1 < argc; ++i) {
        for (const char * n : names) {
            if (std::strcmp(argv[i], n) == 0) {
                return argv[i + 1];
            }
        }
    }
    return "";
}

bool has_flag(int argc, char ** argv, const char * name) {
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) {
            return true;
        }
    }
    return false;
}

// Whether, and how, to stream the model's experts: an explicit mode, or (auto) Kera's cache when the model
// doesn't fit in the memory the machine profile allows. Returns false if nothing is to be streamed.
bool stream_config(const std::string & model, const std::string & mode_arg, double budget_gb_arg, stream::config & sc,
                   std::string & why) {
    stream::cache_mode mode = stream::cache_mode::off;
    const double setting = kr_setting_get("stream.mode");
    const bool automatic = mode_arg.empty() ? setting < -0.5 : mode_arg == "auto";
    if (!mode_arg.empty() && !automatic && !stream::parse_mode(mode_arg, mode)) {
        why = "--stream takes auto, off, os or kera";
        return false;
    }
    if (mode_arg.empty() && !automatic) {
        mode = setting > 1.5 ? stream::cache_mode::kera : setting > 0.5 ? stream::cache_mode::os : stream::cache_mode::off;
    }
    model_sizes m;
    std::string err;
    const bool sized = read_model_sizes(model, m, &err);
    machine_profile prof;
    const bool profiled = load_profile(prof, profile_path());
    double cache_gb = 0;
    if (sized && profiled) {
        const plan_result r = plan(m, plan_input_from_profile(prof));
        if (automatic) {
            if (r.fits || !m.moe()) {
                return false; // it fits (or isn't a MoE): nothing to stream
            }
            mode = stream::cache_mode::kera;
            why = m.file + " doesn't fit in " + std::to_string((int) prof.usable_gb) + " GB of memory";
        }
        cache_gb = r.expert_cache_gb;
    } else if (automatic) {
        return false; // without a profile there's no basis for streaming
    }
    if (mode == stream::cache_mode::off) {
        return false;
    }
    sc.model_path = model;
    sc.mode = mode;
    // what's left for experts, less room for the chat memory and the working buffers
    double budget = budget_gb_arg > 0 ? budget_gb_arg : kr_setting_get("stream.budget_gb");
    if (budget <= 0) {
        budget = std::max(1.0, cache_gb * 0.9 - 1.5);
    }
    sc.budget_bytes = (uint64_t) (budget * 1e9);
    sc.lookahead = kr_setting_get("stream.lookahead") > 0.5;
    sc.io_threads = (int) kr_setting_get("stream.io_threads");
    if (why.empty()) {
        why = "asked for";
    }
    return true;
}

// The server's RAM tier for recent chats' states (llama.cpp's --cache-ram, up to 8 GB by default), sized to the memory
// the model leaves; Kera's saved chats on the SSD keep the older ones (4b item 5). 0 keeps llama.cpp's default.
int prompt_cache_mib(const model_sizes & m, bool have_sizes, bool streamed) {
    if (streamed) {
        // the memory the experts don't use is only the margin left for the chat and the working buffers: 8 GB of
        // saved chat states (143 MB each for Nemotron 3 Nano) made a 15 GB machine re-read the model from the SSD
        return 1024;
    }
    machine_profile prof;
    if (!have_sizes || !load_profile(prof, profile_path())) {
        return 0;
    }
    const double free_gb = prof.usable_gb - m.file_gb - 2.0; // less the chat's own memory and the working buffers
    return (int) std::clamp(free_gb * 1024 / 2, 512.0, 8192.0);
}

// Whether the command line sets up guessing itself (a guesser type, a draft model or llama.cpp's preset)
bool sets_guessing(int argc, char ** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--spec", 0) == 0 || a.find("draft") != std::string::npos || a == "-md" || a == "-hfd" ||
            a == "-hfrd") {
            return true;
        }
    }
    return false;
}

// The guessers to use when the command line sets none (spec 11.1, M4): copying from the chat (llama.cpp's
// "ngram-mod" guesser, with its own default settings: it guesses only when the chat already holds a long
// stretch that fits), then the model's own MTP head if the file has one, unless it didn't pay for this
// model last time (the memory). The switch measures whether each pays, and how far to guess, per chat.
std::string default_guessers(const model_sizes & m, const guess_memory & memory, const std::string & key, bool & mtp) {
    mtp = m.has_mtp;
    guess_verdict v;
    if (mtp && memory.get(key, "draft-mtp", v) && !v.pays) {
        mtp = false;
        std::fprintf(stderr, "Kera: leaving the model's MTP head out: last time it didn't pay on this machine "
                             "(over %lld steps); it will be tried again within 14 days\n", v.steps);
    }
    return mtp ? "ngram-mod,draft-mtp" : "ngram-mod";
}

// The model card's family for this file (D-013), unknown where the file sets its own sampling unlike the card's; says
// which it uses
model_family card_family(const model_sizes & ms) {
    model_family fam = family_of(ms.arch, ms.name);
    sampling_values file;
    file.temp = ms.gguf_temp;
    file.top_p = ms.gguf_top_p;
    file.top_k = ms.gguf_top_k;
    file.min_p = ms.gguf_min_p;
    if (fam != model_family::unknown && !card_agrees_with_file(fam, ms.name, file)) {
        std::fprintf(stderr, "Kera: %s's file sets its own sampling, unlike %s's card: using the file's\n",
                     ms.name.c_str(), family_name(fam));
        return model_family::unknown;
    }
    if (fam != model_family::unknown) {
        std::fprintf(stderr, "Kera: using %s's recommended sampling where neither a request nor the command line "
                             "sets one (--no-model-defaults turns this off)\n", family_name(fam));
    }
    return fam;
}

// A model's own default sampling from its makers' generation_config.json (tools/cards, D-036): for a file that sets
// none of its own and isn't one of the cards read by hand. Says which it uses.
bool listed_defaults(const model_sizes & ms, const std::string & path, sampling_values & out) {
    if (ms.gguf_any_sampling) {
        return false; // the file carries its makers' values already (llama.cpp reads them)
    }
    std::vector<std::string> names = { ms.name };
    if (!ms.basename.empty()) {
        names.push_back(ms.basename + " " + ms.size_label + " " + ms.finetune);
    }
    std::string stem;
    precision_of(std::filesystem::u8path(path).filename().u8string(), &stem);
    names.push_back(stem);
    std::string repo;
    if (!listed_sampling(names, ms.n_layer, ms.n_embd, out, repo)) {
        return false;
    }
    std::string what;
    char buf[48];
    const auto add = [&](const char * label, double v, bool whole) {
        if (v >= 0) {
            if (whole) {
                std::snprintf(buf, sizeof(buf), "%s %.0f", label, v);
            } else {
                std::snprintf(buf, sizeof(buf), "%s %g", label, v);
            }
            what += (what.empty() ? "" : ", ") + std::string(buf);
        }
    };
    add("temperature", out.temp, false);
    add("top_p", out.top_p, false);
    add("top_k", out.top_k, true);
    add("min_p", out.min_p, false);
    add("repetition penalty", out.repeat, false);
    std::fprintf(stderr, "Kera: using the model's own default sampling (%s's generation_config.json: %s) where neither a "
                         "request nor the command line sets one (--no-model-defaults turns this off)\n",
                 repo.c_str(), what.c_str());
    return true;
}

// The request defaults for a model file: its card's (read by hand), else its makers' listed values, with its
// packaging's over them; null when there is nothing to add
std::unique_ptr<model_request_defaults> make_request_defaults(const model_sizes & ms, const std::string & path,
                                                              bool model_defaults_on, bool have_sizes,
                                                              packaging_defaults pack) {
    model_family fam = model_family::unknown;
    sampling_values listed;
    bool has_listed = false;
    if (model_defaults_on && have_sizes) {
        fam = card_family(ms);
        has_listed = fam == model_family::unknown && listed_defaults(ms, path, listed);
    }
    if (fam == model_family::unknown && !has_listed && !pack.any_sampling && pack.system.empty() &&
        pack.messages.empty()) {
        return nullptr;
    }
    return std::make_unique<model_request_defaults>(fam, ms.name, std::move(pack),
                                                    recommended_presence_penalty(fam, ms.name, ms.quantized()), listed,
                                                    has_listed);
}

// the image reader Kera found for the model (D-027), "" if none
std::string g_projector;

// a model llama.cpp finds by itself (-hf, LLAMA_ARG_MODEL, a preset) gets the card's sampling too, once its file is
// known (the owner's choice, 9 Nov): set where no -m names the model and the card's sampling is on
bool g_card_late = false;
std::unique_ptr<model_request_defaults> g_late_defaults;

// hook 8: once llama.cpp has read every source of its settings (the command line, LLAMA_ARG_* variables, config.ini,
// presets), Kera's reader is used only where none of them names one or turns it off; and a model that reads images
// keeps its chats off the SSD (as before, D-027), whichever source named its reader
void kera_fill_params(common_params & p) {
    static bool told = false; // the server loads the model again after sleeping
    if (!g_projector.empty() && p.mmproj.path.empty() && p.mmproj.url.empty() && !p.no_mmproj &&
        p.model.hf_repo.empty()) {
        p.mmproj.path = g_projector;
        if (!told) {
            std::fprintf(stderr, "Kera: the model can see images (its image reader: %s; --no-mmproj turns this off)\n",
                         g_projector.c_str());
        }
    } else if (!g_projector.empty() && !told) {
        std::fprintf(stderr, "Kera: not adding %s: your settings %s\n", g_projector.c_str(),
                     p.no_mmproj ? "turn the image reader off" : "name the model's image reader");
    }
    told = true;
    if (!p.mmproj.path.empty() && common_get_prompt_store() != nullptr) {
        common_set_prompt_store(nullptr);
    }
    if (g_card_late && common_get_request_defaults() == nullptr && !p.model.path.empty()) {
        g_card_late = false;
        model_sizes ms;
        if (read_model_sizes(p.model.path, ms)) {
            g_late_defaults = make_request_defaults(ms, p.model.path, true, true, packaging_defaults{});
            if (g_late_defaults) {
                common_set_request_defaults(g_late_defaults.get());
            }
        }
    }
}

} // namespace

int run_serve(int argc, char ** argv) {
    kr_settings_register_builtin();
#if !defined(_WIN32)
    // an engine of kera serve --backend auto stops when that kera serve is gone, however it went (D-024)
    if (const char * pp = std::getenv("KERA_PARENT_PID"); pp != nullptr) {
        const pid_t parent = (pid_t) std::atoll(pp);
        unsetenv("KERA_PARENT_PID");
        std::thread([parent] {
            while (getppid() == parent) {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
            kill(getpid(), SIGTERM);
        }).detach();
    }
#endif
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        std::printf(
            "kera serve: llama.cpp's server with Kera's saved chats, guessing ahead and expert streaming. Kera's options:\n"
            "  --backend auto|auto-model|vulkan|hip|cpu\n"
            "                                 the graphics back-end; auto measures Vulkan and the ROCm add-on on each model and,\n"
            "                                 with memory for the model twice, runs it on both and sends each message to the one\n"
            "                                 that finishes it sooner; auto-model (or too little memory): the faster for how the\n"
            "                                 model is used\n"
            "  --multi-user                   for several people at once: a new message read first, long prompts giving way\n"
            "                                 to writing, a start several chats share read once, request priorities, /metrics\n"
            "  --stream auto|off|os|kera      stream a MoE model's experts from the SSD (auto: when it doesn't fit)\n"
            "  --stream-budget-gb N           RAM for Kera's expert cache\n"
            "  --no-kv-store                  don't keep chats on the SSD\n"
            "  --no-auto-guess                don't guess ahead unless llama.cpp's own options ask for it\n"
            "  --no-guess-controller          llama.cpp's own guessing lengths instead of Kera's measured ones\n"
            "  --no-model-defaults            the server's sampling instead of the model card's\n"
            "\nllama.cpp's options follow.\n\n");
        std::fflush(stdout);
    }
    // music models (a folder of them) are served by the music engine (D-028)
    if (has_flag(argc, argv, "--music")) {
        return run_music_server(argc, argv);
    }
    if (has_flag(argc, argv, "--sfx")) {
        return run_sfx_server(argc, argv);
    }
    // an image model (or its parts named on the command line) is served by the image engine instead (D-027)
    {
        const std::string m = flag_value(argc, argv, { "-m", "--model" });
        if (has_flag(argc, argv, "--diffusion-model") || (!m.empty() && is_image_model(m))) {
            return run_image_server(argc, argv);
        }
    }
    // Kera's own option, taken out before llama.cpp's server reads the rest
    std::vector<char *> args;
    bool store_on = kr_setting_get("kv.store") > 0.5;
    bool guess_on = kr_setting_get("spec.controller") > 0.5;
    std::string stream_mode;
    double stream_budget_gb = 0;
    bool model_defaults_on = kr_setting_get("chat.model_defaults") > 0.5;
    bool auto_guess = kr_setting_get("spec.auto") > 0.5;
    bool short_first = kr_setting_get("serve.short_first") > 0.5;
    double read_share = kr_setting_get("serve.read_share");
    bool share_prefix = kr_setting_get("serve.share_prefix") > 0.5;
    bool metrics = kr_setting_get("serve.metrics") > 0.5;
    std::string backend;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-kv-store") == 0) {
            store_on = false;
            continue;
        }
        if (std::strcmp(argv[i], "--no-model-defaults") == 0) {
            model_defaults_on = false;
            continue;
        }
        if (std::strcmp(argv[i], "--stream") == 0 && i + 1 < argc) {
            stream_mode = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--stream-budget-gb") == 0 && i + 1 < argc) {
            stream_budget_gb = std::atof(argv[++i]);
            continue;
        }
        if (std::strcmp(argv[i], "--no-guess-controller") == 0) {
            guess_on = false;
            continue;
        }
        if (std::strcmp(argv[i], "--no-auto-guess") == 0) {
            auto_guess = false;
            continue;
        }
        if (std::strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            backend = argv[++i];
            continue;
        }
        if (std::strcmp(argv[i], "--multi-user") == 0) {
            short_first = true;
            read_share = read_share > 0 ? read_share : 0.5;
            share_prefix = true;
            metrics = true;
            continue;
        }
        args.push_back(argv[i]);
    }
    // only matters when guessing is set up (a draft model, MTP, or n-gram guessing)
    guess_controller guesser;
    if (guess_on) {
        common_set_spec_controller(&guesser);
    }
    const int n = (int) args.size();

    const std::string model = flag_value(n, args.data(), { "-m", "--model" });
    note_higher_precision(model, "-m");

    // a vision model's image reader, found next to the model or in Ollama's store (D-027), used only where no source of
    // llama.cpp's settings (the command line and its aliases, LLAMA_ARG_* variables, config.ini) names one or turns it
    // off: llama.cpp fills it in once it has read them all (hook 8, kera_fill_params)
    static std::string projector;
    if (!model.empty() && !has_flag(n, args.data(), "--mmproj") && !has_flag(n, args.data(), "-mm") &&
        !has_flag(n, args.data(), "--mmproj-url") && !has_flag(n, args.data(), "-mmu") &&
        !has_flag(n, args.data(), "--no-mmproj") && !has_flag(n, args.data(), "--no-mmproj-auto") &&
        !has_flag(n, args.data(), "-hf") && !has_flag(n, args.data(), "-hfr") && !has_flag(n, args.data(), "--hf-repo")) {
        std::string left_out;
        projector = find_projector(model, &left_out);
        if (!left_out.empty()) {
            std::fprintf(stderr, "Kera: found %s next to the model, but nothing says it is this model's image reader, so "
                                 "the model stays text-only (--mmproj %s uses it)\n", left_out.c_str(), left_out.c_str());
        }
        g_projector = projector;
    }
    common_set_params_fill(kera_fill_params);

    // a model from Ollama's store runs the way Ollama runs it (rule one, D-033): never in a chat format it wasn't trained
    // on, and with its Modelfile's adapters (its settings and messages are applied with the card's below)
    static ollama_model from_ollama;
    static std::vector<std::string> ollama_adapters;
    static char lora_flag[] = "--lora";
    const bool is_ollama = !model.empty() && ollama_model_of(model, from_ollama);
    if (is_ollama) {
        const bool own_template = has_flag(n, args.data(), "--chat-template") ||
                                  has_flag(n, args.data(), "--chat-template-file");
        if (!own_template && !has_chat_template(model)) {
            // llama.cpp would quietly fall back to ChatML; the model would be prompted in a format it wasn't trained on
            std::fprintf(stderr,
                         "kera serve: %s, from Ollama's store, keeps its chat format in Ollama's own form, which Kera "
                         "can't read, and its file has none. Without it the model would be prompted in a format it wasn't "
                         "trained on and answer worse. Run it in Ollama, use its maker's GGUF file (which carries the "
                         "format), or give it with --chat-template-file.\n", from_ollama.name.c_str());
            return 2;
        }
        ollama_adapters = from_ollama.adapters;
        for (auto & a : ollama_adapters) {
            args.push_back(lora_flag);
            args.push_back(a.data());
        }
        if (!ollama_adapters.empty()) {
            std::fprintf(stderr, "Kera: %s's Modelfile adds %zu adapter(s), as in Ollama\n", from_ollama.name.c_str(),
                         ollama_adapters.size());
        }
    }

    // the graphics back-end: when asked for (the optional ROCm add-on, spec Phase 8), or the faster of Vulkan and ROCm
    // for this model and how it is used (--backend auto, or the setting gpu.auto_backend; D-022). A model whose experts
    // are streamed keeps llama.cpp's choice: the SSD sets its speed, which kera calibrate doesn't measure.
    // With room for the model twice, both back-ends run it and each message goes to the one that finishes it sooner
    // (D-024); --backend auto-model keeps to one back-end for the model.
    bool backend_was_auto = false;
    if (backend == "auto" || backend == "auto-model" || (backend.empty() && kr_setting_get("gpu.auto_backend") > 0.5)) {
        const bool asked = backend == "auto" || backend == "auto-model";
        stream::config probe;
        std::string why;
        const bool streamed = !model.empty() && stream_config(model, stream_mode, stream_budget_gb, probe, why);
        if (!streamed && backend != "auto-model") {
            const int rc = serve_each_message(model, argc, argv, (int) args.size(), args.data());
            if (rc >= 0) {
                return rc;
            }
        }
        backend = streamed ? "" : auto_backend(model, (int) args.size(), args.data());
        backend_was_auto = true;
        if (asked && backend.empty() && (streamed || !hip_addon_installed())) {
            std::fprintf(stderr, "Kera: --backend auto: %s, so the back-end is llama.cpp's choice\n",
                         streamed ? "the model's experts are streamed from the SSD" : "the ROCm add-on isn't installed");
        }
    }
    static std::string device;
    static char device_flag[] = "--device";
    {
        std::string err;
        bool chosen = choose_backend(backend, device, &err);
        if (!chosen && backend_was_auto && !backend.empty()) {
            // measured faster once, but it doesn't load now (a driver update, the add-on removed): llama.cpp's
            // own choice rather than no model at all
            std::fprintf(stderr, "Kera: %s; using llama.cpp's choice of back-end instead\n", err.c_str());
            backend.clear();
            device.clear();
            err.clear();
            chosen = choose_backend(backend, device, &err);
        }
        if (!chosen) {
            std::fprintf(stderr, "Kera: %s\n", err.c_str());
            return 2;
        }
        if (!device.empty() && !has_flag(n, args.data(), "--device") && !has_flag(n, args.data(), "-dev")) {
            args.push_back(device_flag);
            args.push_back(device.data());
            std::fprintf(stderr, "Kera: using the %s back-end (device %s)\n", backend.c_str(), device.c_str());
        }
    }

    // the model's own sizes and design, for the defaults below
    model_sizes ms;
    const bool have_sizes = !model.empty() && read_model_sizes(model, ms);

    // guessers, when the command line sets none, and what was learned about them for this model
    guess_memory memory(home_path("guessers.json"));
    const std::string model_key = model.empty() ? "" : guess_memory_key(model);
    guesser.remember(&memory, model_key);
    static std::string guess_types;
    static char spec_type_flag[] = "--spec-type";
    if (guess_on && auto_guess && have_sizes && !sets_guessing(n, args.data())) {
        bool mtp = false;
        guess_types = default_guessers(ms, memory, model_key, mtp);
        args.push_back(spec_type_flag);
        args.push_back(guess_types.data());
        std::fprintf(stderr, "Kera: guessing ahead with %s; Kera measures which pays (--no-auto-guess turns this off)\n",
                     mtp ? "copying from the chat and the model's own MTP head" : "copying from the chat");
    }

    auto store = std::make_unique<server_store_adapter>();
    std::atomic<bool> server_running{ true }; // false once llama.cpp's server has stopped
    std::thread hasher;
    const int n_all = (int) args.size(); // with the guessers added above
    if (store_on && !model.empty() && !has_flag(n_all, args.data(), "--mmproj")) {
        // each back-end's sums differ in their last bits: a chat saved on one is read again on another, not continued
        // there, so its answers stay the ones that back-end gives (Vulkan's states keep their old folder)
        std::string extra;
        if (backend == "hip" || backend == "cpu") {
            extra += "|backend=" + backend;
        }
        const uint64_t budget = (uint64_t) (kr_setting_get("kv.disk_budget_gb") * 1e9);
        common_set_prompt_store(store.get());
        hasher = std::thread([&store, &server_running, model, extra, budget] {
            // The cache identity: everything that decides the numbers a saved chat's memory holds, so a chat is only
            // ever continued from memory this engine, model and settings would compute (rule one, D-033):
            // - this build of Kera and llama.cpp: an update may change how a model is computed;
            // - every part of the model file, by content;
            // - the settings as the server parsed them, wherever they came from (command line, LLAMA_ARG_* variables,
            //   a preset): rope and YaRN, the memory's types, flash attention, model setting overrides, adapters and
            //   control vectors with their scales, the guessers, and whether a LongRoPE model uses its long factors
            //   (common/prompt-store.h, set_identity);
            // - the contents of the files those settings name (adapters, control vectors, a draft model).
            // The server says once it has loaded the model, so the first hashing of a big model doesn't share the
            // drive with the load (0.9 s more on a 4 GB file, about 10 s on the Minisforum's 23 GB one). Saved chats
            // can't be found until then, and none are wanted before.
            std::string settings;
            std::vector<std::string> files;
            if (!store->wait_identity(server_running, settings, files)) {
                return;
            }
            int err = 0;
            std::string ident = "kera " KERA_VERSION " " KERA_GIT_COMMIT "|llama.cpp " KERA_LLAMA_CPP_TAG;
            for (const auto & part : model_parts(model)) {
                const std::string h = model_hash(part, &err);
                if (h.empty()) {
                    std::fprintf(stderr, "Kera: can't hash %s (error %d); chats won't be saved to the SSD\n", part.c_str(), err);
                    return;
                }
                ident += "|" + h;
            }
            ident += "|" + settings + extra;
            for (const auto & f : files) {
                const std::string h = model_hash(f, &err);
                if (h.empty()) {
                    std::fprintf(stderr, "Kera: can't hash %s (error %d); chats won't be saved to the SSD\n", f.c_str(), err);
                    return;
                }
                ident += "|file=" + h;
            }
            const std::string dir = join(home_path("kvstore"), hash_bytes(ident.data(), ident.size()).substr(0, 16));
            std::error_code dir_ec;
            std::filesystem::create_directories(std::filesystem::u8path(dir), dir_ec);
            const uint64_t others = trim_saved_prompts(home_path("kvstore"), dir, budget);
            store->open(dir, budget > others ? budget - others : budget / 2);
            std::fprintf(stderr, "Kera: saving chat states in %s (up to %.0f GB)\n", dir.c_str(), budget / 1e9);
        });
    }

    // the model card's sampling for thinking, no thinking and tool calls, where neither a request nor the command line
    // sets it; only for the card's own model (by design and name, and the file's own defaults are one of the card's)
    std::unique_ptr<model_request_defaults> defaults;
    g_card_late = model_defaults_on && model.empty(); // -hf, LLAMA_ARG_MODEL: once llama.cpp has the file (hook 8)
    // an Ollama model's Modelfile: its SYSTEM text and MESSAGE pairs always, its PARAMETERs unless --no-model-defaults
    packaging_defaults pack;
    if (is_ollama) {
        pack = ollama_packaging(from_ollama, model_defaults_on);
        if (pack.any_sampling || !pack.system.empty() || !pack.messages.empty()) {
            std::fprintf(stderr, "Kera: using %s's Modelfile, as Ollama does:%s%s%s\n", from_ollama.name.c_str(),
                         pack.any_sampling ? " its settings" : "", pack.system.empty() ? "" : " its system prompt",
                         pack.messages.empty() ? "" : " its example messages");
        }
    }
    defaults = make_request_defaults(ms, model, model_defaults_on, have_sizes, std::move(pack));
    if (defaults) {
        common_set_request_defaults(defaults.get());
    }

    // stream the experts of a MoE model that doesn't fit (M5)
    std::shared_ptr<stream::session> streaming;
    std::atomic<bool> serving{ true };
    std::thread reporter;
    if (!model.empty()) {
        stream::config sc;
        std::string why;
        if (stream_config(model, stream_mode, stream_budget_gb, sc, why)) {
            std::string err;
            streaming = stream::start(sc, &err);
            if (streaming) {
                // read prompts in big batches: each batch runs layer by layer, so each expert comes off the
                // SSD at most once per batch, not once per small chunk (spec 4b item 3; RT2: a prompt uses
                // nearly every expert)
                static char ub_flag[] = "-ub";
                static char ub_val[] = "2048";
                static char b_flag[] = "-b";
                if (!has_flag(n, args.data(), "-ub") && !has_flag(n, args.data(), "--ubatch-size")) {
                    args.push_back(ub_flag);
                    args.push_back(ub_val);
                    if (!has_flag(n, args.data(), "-b") && !has_flag(n, args.data(), "--batch-size")) {
                        args.push_back(b_flag);
                        args.push_back(ub_val);
                    }
                }
                // every extra word a guess check reads brings its own experts from the SSD (RT2: 4 words use 2.9x
                // one word's), so guessers start low; the switch still measures upward from there
                guess_switch::options go;
                go.model_start = 1; // k = 1
                go.free_start = 1;  // k = 4
                guesser.set_options(go);
                if (sc.mode == stream::cache_mode::kera) {
                    std::fprintf(stderr, "Kera: streaming the model's experts from the SSD (%s), with %.1f GB of RAM for them\n",
                                 why.c_str(), (double) sc.budget_bytes / 1e9);
                } else {
                    std::fprintf(stderr, "Kera: streaming the model's experts from the SSD (%s), through the OS file cache\n",
                                 why.c_str());
                }
                // every minute with activity: what streaming did, per word
                reporter = std::thread([&] {
                    stream::report last = streaming->snapshot();
                    auto next = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                    while (serving) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        if (std::chrono::steady_clock::now() < next) {
                            continue;
                        }
                        next = std::chrono::steady_clock::now() + std::chrono::seconds(60);
                        if (!streaming->streaming()) {
                            continue;
                        }
                        const stream::report now = streaming->snapshot();
                        const stream::report d = now - last;
                        last = now;
                        const auto layers = std::max<uint64_t>(1, streaming->moe_layers());
                        if (d.steps >= layers) {
                            std::string line = stream::session::per_word(d, d.steps / layers);
                            if (const double b = streaming->predicted_new_bytes(3); b > 0) {
                                char buf[96];
                                std::snprintf(buf, sizeof(buf), "; checking 3 guesses would read about %.0f MB from the SSD", b / 1e6);
                                line += buf;
                            }
                            std::fprintf(stderr, "Kera: last minute, %s\n", line.c_str());
                        }
                    }
                });
            } else {
                std::fprintf(stderr, "Kera: can't stream the experts (%s); loading the model as usual\n", err.c_str());
            }
        } else if (!why.empty()) {
            std::fprintf(stderr, "Kera: %s\n", why.c_str());
            common_set_prompt_store(nullptr);
            server_running = false;
            if (hasher.joinable()) {
                hasher.join(); // a thread still running would end the program when it goes out of scope
            }
            return 2;
        }
    }

    // several chats at once (D-023): a new message first, prompt reading giving way to writing, shared starts
    std::unique_ptr<share_scheduler> sharing;
    if (streaming && read_share > 0) {
        // each prompt step reads the experts it uses from the SSD, so short steps would read them many times over
        read_share = 0;
        std::fprintf(stderr, "Kera: the experts are streamed, so long prompts keep whole steps while others write\n");
    }
    if (short_first || read_share > 0 || share_prefix) {
        chat_share_options o;
        o.short_first = short_first;
        o.read_share = read_share;
        o.step_s = kr_setting_get("serve.read_step_s");
        sharing = std::make_unique<share_scheduler>(o, share_prefix ? 64 : 0);
        common_set_batch_scheduler(sharing.get());
        std::string what;
        if (short_first) {
            what += "the prompt with the fewest words left is read first";
        }
        if (read_share > 0) {
            char buf[160];
            std::snprintf(buf, sizeof(buf), "%sa long prompt takes at most %.0f%% of the chip's time while others write, "
                                            "in steps of about %.1f s", what.empty() ? "" : "; ", 100 * read_share, o.step_s);
            what += buf;
        }
        if (share_prefix) {
            what += std::string(what.empty() ? "" : "; ") + "a start two chats have in common is read once";
        }
        std::fprintf(stderr, "Kera: several chats at once: %s\n", what.c_str());
    }
    static char metrics_flag[] = "--metrics";
    if (metrics && !has_flag(n, args.data(), "--metrics")) {
        args.push_back(metrics_flag);
    }
    // sharing a start needs one memory for all chats: llama.cpp's choice when it picks the number of chats, but not
    // when -np sets it
    static char kvu_flag[] = "-kvu";
    if (share_prefix && (has_flag(n, args.data(), "-np") || has_flag(n, args.data(), "--parallel")) &&
        !has_flag(n, args.data(), "-kvu") && !has_flag(n, args.data(), "--kv-unified") &&
        !has_flag(n, args.data(), "-no-kvu") && !has_flag(n, args.data(), "--no-kv-unified")) {
        args.push_back(kvu_flag);
    }

    // recent chats' states in RAM: as much as the model leaves room for (llama.cpp's own default is 8 GB)
    static std::string cram_val;
    static char cram_flag[] = "--cache-ram";
    if (!has_flag(n, args.data(), "-cram") && !has_flag(n, args.data(), "--cache-ram")) {
        if (const int mib = prompt_cache_mib(ms, have_sizes, streaming != nullptr); mib > 0) {
            cram_val = std::to_string(mib);
            args.push_back(cram_flag);
            args.push_back(cram_val.data());
        }
    }

    // an NVIDIA chip sharing the computer's memory (DGX Spark): the model is read into memory, not memory-mapped
    // (spec 10.6: 56.3 against 44.7 tok/s measured on a Spark; the setting gpu.shared_no_mmap). Streamed experts
    // keep llama.cpp's default: Kera's cache reads them itself.
    static char lm_flag[] = "-lm";
    static char lm_none[] = "none";
    if (streaming == nullptr && kr_setting_get("gpu.shared_no_mmap") > 0.5 && !has_flag(n, args.data(), "-lm") &&
        !has_flag(n, args.data(), "--load-mode") && !has_flag(n, args.data(), "--no-mmap") &&
        !has_flag(n, args.data(), "--mmap")) {
        machine_profile prof;
        if (load_profile(prof, profile_path()) && !prof.gpus.empty() && prof.gpus[0].backend == "CUDA" &&
            prof.gpus[0].unified) {
            args.push_back(lm_flag);
            args.push_back(lm_none);
            std::fprintf(stderr, "Kera: %s shares this computer's memory, so the model is read in whole rather than "
                                 "memory-mapped (faster on a DGX Spark)\n", prof.gpus[0].name.c_str());
        }
    }

    // each reply's shape, for the next start's choice of back-end (hook 6)
    note_replies(model, backend, (int) args.size(), args.data());
    const int rc = llama_server((int) args.size(), args.data());
    server_running = false;
    stop_noting_replies();
    serving = false;
    if (reporter.joinable()) {
        reporter.join();
    }
    if (streaming) {
        // the whole run, in the same words as the per-minute lines (the model is freed by now; the counts stay)
        const stream::report r = streaming->snapshot();
        const auto layers = std::max<uint64_t>(1, streaming->moe_layers());
        if (r.steps >= layers) {
            std::fprintf(stderr, "Kera: over this run, %s\n", stream::session::per_word(r, r.steps / layers).c_str());
        }
    }
    stream::stop();
    common_set_batch_scheduler(nullptr);
    common_set_request_defaults(nullptr);
    common_set_spec_controller(nullptr);
    common_set_prompt_store(nullptr);
    if (hasher.joinable()) {
        hasher.join();
    }
    store.reset(); // writes what's still queued
    return rc;
}

// kera cli: llama.cpp's own chat, which runs llama.cpp's server inside it, with the model card's sampling where neither
// a message nor the command line sets one, as kera serve gives it (the owner's choice, 9 Nov; D-013)
int run_cli(int argc, char ** argv) {
    bool model_defaults_on = kr_setting_get("chat.model_defaults") > 0.5;
    std::vector<char *> args;
    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-model-defaults") == 0) {
            model_defaults_on = false;
            continue;
        }
        args.push_back(argv[i]);
    }
    const int n = (int) args.size();
    const std::string model = flag_value(n, args.data(), { "-m", "--model" });
    note_higher_precision(model, "-m");
    std::unique_ptr<model_request_defaults> defaults;
    model_sizes ms;
    if (model_defaults_on && !model.empty() && read_model_sizes(model, ms)) {
        defaults = make_request_defaults(ms, model, true, true, packaging_defaults{});
        if (defaults) {
            common_set_request_defaults(defaults.get());
        }
    }
    g_card_late = model_defaults_on && model.empty(); // -hf, LLAMA_ARG_MODEL: once llama.cpp has the file (hook 8)
    common_set_params_fill(kera_fill_params);
    const int rc = llama_cli(n, args.data());
    common_set_params_fill(nullptr);
    common_set_request_defaults(nullptr);
    return rc;
}

} // namespace kera
