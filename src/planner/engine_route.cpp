// Which back-end serves each message (engine_route.h).

#include "planner/engine_route.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <limits>

namespace kera {

namespace {

const size_t k_piece = 1024;

uint64_t fnv(uint64_t h, const char * p, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        h ^= (unsigned char) p[i];
        h *= 1099511628211ull;
    }
    return h;
}

struct piece_builder {
    std::vector<route_piece> out;
    uint64_t chain = 14695981039346656037ull;

    // one message (or the system prompt, or the tools): its role first, then its text in stretches of k_piece
    void add(const std::string & role, const std::string & text, bool reply) {
        const std::string head = "\x1e" + role + "\x1f";
        chain = fnv(chain, head.data(), head.size());
        if (text.empty()) {
            out.push_back({ chain, (double) head.size(), reply });
            return;
        }
        for (size_t at = 0; at < text.size(); at += k_piece) {
            const size_t n = std::min(k_piece, text.size() - at);
            chain = fnv(chain, text.data() + at, n);
            out.push_back({ chain, (double) (n + (at == 0 ? head.size() : 0)), reply });
        }
    }
};

// a message's text: a string, or the text of its parts (anything else as its JSON)
std::string text_of(const nlohmann::json & content) {
    if (content.is_string()) {
        return content.get<std::string>();
    }
    if (content.is_null()) {
        return "";
    }
    if (content.is_array()) {
        std::string s;
        for (const auto & part : content) {
            if (part.is_object() && part.contains("text") && part["text"].is_string()) {
                s += part["text"].get<std::string>();
            } else {
                s += part.dump();
            }
        }
        return s;
    }
    return content.dump();
}

} // namespace

std::vector<route_piece> route_pieces(const std::string & body, double * written) {
    if (written) {
        *written = 0;
    }
    const nlohmann::json j = nlohmann::json::parse(body, nullptr, false);
    if (j.is_discarded() || !j.is_object()) {
        return {};
    }
    if (written) {
        for (const char * k : { "max_tokens", "max_completion_tokens", "max_output_tokens", "n_predict" }) {
            if (j.contains(k) && j[k].is_number() && j[k].get<double>() > 0) {
                *written = j[k].get<double>();
                break;
            }
        }
    }
    piece_builder b;
    if (j.contains("tools") && !j["tools"].is_null()) {
        b.add("tools", j["tools"].dump(), false);
    }
    if (j.contains("system")) { // Anthropic's
        b.add("system", text_of(j["system"]), false);
    }
    if (j.contains("instructions") && j["instructions"].is_string()) { // OpenAI's Responses
        b.add("system", j["instructions"].get<std::string>(), false);
    }
    const nlohmann::json * msgs = nullptr;
    if (j.contains("messages") && j["messages"].is_array()) {
        msgs = &j["messages"];
    } else if (j.contains("input") && j["input"].is_array()) {
        msgs = &j["input"];
    }
    if (msgs != nullptr) {
        for (const auto & m : *msgs) {
            if (!m.is_object()) {
                b.add("", m.dump(), false);
                continue;
            }
            const std::string role = m.contains("role") && m["role"].is_string() ? m["role"].get<std::string>() : "";
            std::string text = m.contains("content") ? text_of(m["content"]) : text_of(m);
            if (m.contains("tool_calls") && !m["tool_calls"].is_null()) {
                text += m["tool_calls"].dump();
            }
            b.add(role, text, role == "assistant");
        }
        return b.out;
    }
    for (const char * k : { "prompt", "input" }) {
        if (j.contains(k)) {
            b.add("prompt", text_of(j[k]), false);
            return b.out;
        }
    }
    return {};
}

engine_router::engine_router(std::vector<backend_speed> engines, engine_route_options o)
    : o_(o), bytes_per_token_(o.bytes_per_token), reply_tokens_(o.reply_tokens) {
    for (auto & s : engines) {
        engine e;
        e.speed = s;
        e.load_s = o_.load_s;
        engines_.push_back(std::move(e));
    }
}

void engine_router::set_ready(int e, bool r) {
    engines_[e].ready = r;
    if (r && o_.one_at_a_time) {
        for (auto & x : engines_) {
            x.saved_s = 0;
        }
    }
}

void engine_router::set_failed(int e) {
    engines_[e].failed = true;
    engines_[e].ready = false;
}

void engine_router::set_load_s(int e, double s) {
    engines_[e].load_s = s;
}

void engine_router::forget(int e) {
    engines_[e].held.clear();
}

int engine_router::running(int e) const {
    int n = 0;
    for (const auto & [id, r] : running_) {
        n += r.engine == e ? 1 : 0;
    }
    return n;
}

double engine_router::held_bytes(const engine & e, const std::vector<route_piece> & prompt) const {
    // the last piece this engine has seen: every piece before it has the same start, so it holds all of them
    int last = -1;
    double held = 0;
    double at = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        at += prompt[i].bytes;
        if (e.held.count(prompt[i].chain) != 0) {
            last = (int) i;
            held = at;
        }
    }
    // the end of a request it served, then an assistant's message: its own reply, which it holds too
    if (last >= 0 && e.held.at(prompt[last].chain).last) {
        for (size_t i = (size_t) last + 1; i < prompt.size() && prompt[i].reply; ++i) {
            held += prompt[i].bytes;
        }
    }
    return held;
}

double engine_router::busy_s(int e, double now_s) const {
    double s = 0;
    for (const auto & [id, r] : running_) {
        if (r.engine == e) {
            s += std::max(0.0, r.expected_s - (now_s - r.start_s));
        }
    }
    return s;
}

engine_router::choice engine_router::pick(const std::vector<route_piece> & prompt, double max_written, double now_s) const {
    choice c;
    double bytes = 0;
    for (const auto & p : prompt) {
        bytes += p.bytes;
    }
    c.read = std::max(1.0, bytes / bytes_per_token_);
    c.written = max_written > 0 ? std::min(max_written, reply_tokens_) : reply_tokens_;
    // one at a time: the loaded engine, which another waits for before it is loaded
    int loaded = -1;
    for (size_t i = 0; i < engines_.size(); ++i) {
        loaded = engines_[i].ready ? (int) i : loaded;
    }
    const double drain = loaded >= 0 ? busy_s(loaded, now_s) : 0;
    double best = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < engines_.size(); ++i) {
        const engine & e = engines_[i];
        const double held = std::min(c.read - 1, held_bytes(e, prompt) / bytes_per_token_);
        c.held.push_back(std::max(0.0, held));
        c.if_loaded.push_back(e.speed.ok() && !e.failed ? e.speed.reply_seconds(c.held.back(), c.read - c.held.back(), c.written) : -1);
        const bool can_load = o_.one_at_a_time && !e.ready && !e.failed && loaded >= 0;
        if ((!e.ready && !can_load) || (!e.speed.ok() && !e.ready)) {
            c.seconds.push_back(-1);
            continue;
        }
        if (!e.speed.ok()) {
            c.seconds.push_back(-1);
            if (c.engine < 0) {
                c.engine = (int) i; // ready but not measured: only when no measured one is
            }
            continue;
        }
        const double t = e.ready ? c.if_loaded.back() + o_.busy_weight * busy_s((int) i, now_s)
                                 : drain + e.load_s + c.if_loaded.back();
        c.seconds.push_back(t);
        if (t < best) {
            best = t;
            c.engine = (int) i;
        }
    }
    // the time another would have saved since the last change has paid for loading it: when the loaded one is idle,
    // and this request is sooner there too
    if (o_.one_at_a_time && c.engine == loaded && loaded >= 0 && drain <= 0) {
        for (size_t i = 0; i < engines_.size(); ++i) {
            const engine & e = engines_[i];
            if ((int) i != loaded && c.seconds[i] > 0 && e.saved_s >= e.load_s && c.if_loaded[i] < c.if_loaded[loaded]) {
                c.engine = (int) i;
                break;
            }
        }
    }
    c.load = c.engine >= 0 && !engines_[c.engine].ready;
    return c;
}

void engine_router::chosen(const choice & c) {
    if (!o_.one_at_a_time || c.engine < 0 || c.load || c.if_loaded[c.engine] < 0) {
        return;
    }
    for (size_t i = 0; i < engines_.size(); ++i) {
        if ((int) i != c.engine && c.if_loaded[i] >= 0) {
            engines_[i].saved_s += std::max(0.0, c.if_loaded[c.engine] - c.if_loaded[i]);
        }
    }
}

int64_t engine_router::started(int e, const std::vector<route_piece> & prompt, double expected_s, double now_s) {
    auto & held = engines_[e].held;
    ++clock_;
    double at = 0;
    for (size_t i = 0; i < prompt.size(); ++i) {
        at += prompt[i].bytes;
        held_piece & h = held[prompt[i].chain];
        h.end_bytes = at;
        h.used = clock_;
        h.last = h.last || i + 1 == prompt.size();
    }
    if (held.size() > o_.remembered) {
        // forget the half sent longest ago
        std::vector<uint64_t> used;
        used.reserve(held.size());
        for (const auto & [k, h] : held) {
            used.push_back(h.used);
        }
        std::nth_element(used.begin(), used.begin() + used.size() / 2, used.end());
        const uint64_t cut = used[used.size() / 2];
        for (auto it = held.begin(); it != held.end();) {
            it = it->second.used < cut ? held.erase(it) : std::next(it);
        }
    }
    const int64_t id = next_id_++;
    running_[id] = { e, now_s, expected_s };
    return id;
}

void engine_router::finished(int64_t id, double prompt_bytes, double prompt_tokens, double written_tokens) {
    running_.erase(id);
    // only long prompts teach the token size: a short one is mostly the chat template's own tokens
    if (prompt_bytes >= 2048 && prompt_tokens > 0) {
        bytes_per_token_ = std::clamp(0.8 * bytes_per_token_ + 0.2 * prompt_bytes / prompt_tokens, 1.5, 8.0);
    }
    if (written_tokens > 0) {
        reply_tokens_ = 0.9 * reply_tokens_ + 0.1 * written_tokens;
    }
}

} // namespace kera
