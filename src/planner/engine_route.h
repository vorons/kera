// Which back-end serves each message (the owner's request of 18 Oct, docs/DECISIONS.md D-024). On the Minisforum,
// Vulkan writes about a fifth faster while ROCm reads a long document twice as fast (docs/BENCHMARKS.md, 16 Oct), so
// no one back-end is fastest for every message. kera serve sends each request to the engine expected to finish it
// sooner: its measured speeds (backend_pick.h), how much of the request that engine already holds (a follow-up on a
// document stays where the document was read), and the work it has in hand. When memory holds the model twice both
// engines stay loaded; when it holds it once, one is loaded at a time, and the other is loaded in its place (when the
// loaded one is idle) if a request is expected to finish sooner even counting the loading, or once the time the
// other would have saved since the last change adds up to its loading time (a long document goes to ROCm at once; a
// run of chats brings Vulkan back after a while).
#pragma once

#include "planner/backend_pick.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace kera {

// A request's prompt as pieces in order: each message (and the system prompt and tools) cut into stretches of at
// most 1 KB, so a prompt that grows (a raw completion prompt, a long message) still matches what was sent before.
// `chain` is a hash of everything up to and including the piece, so one equal chain means an equal start.
struct route_piece {
    uint64_t chain = 0;
    double bytes = 0;
    bool reply = false; // part of an assistant's message: the engine that wrote it holds it
};

// The pieces of an API request's JSON body (OpenAI chat or completion, Anthropic messages, llama.cpp's /completion);
// empty if the body isn't one, and *written gets the most tokens it asks for (0 if it sets no limit)
std::vector<route_piece> route_pieces(const std::string & body, double * written);

struct engine_route_options {
    double busy_weight = 0.5;        // a running request's remaining seconds count this much against its engine
    double bytes_per_token = 3.6;    // until long prompts teach the real figure
    double reply_tokens = 300;       // a reply's length, until replies teach it
    size_t remembered = 200000;      // pieces remembered per engine
    bool one_at_a_time = false;      // memory for one engine: the other is loaded in its place when that pays
    double load_s = 20;              // an engine's loading time, until it has been measured
};

class engine_router {
public:
    engine_router(std::vector<backend_speed> engines, engine_route_options o = {});

    void set_ready(int engine, bool ready);    // loaded and answering (one at a time: the changes start afresh)
    void set_failed(int engine);               // it couldn't be started: never chosen again
    void set_load_s(int engine, double s);     // how long it took to load, measured
    void forget(int engine);                   // it holds nothing any more (unloaded without saved chats)
    bool ready(int engine) const { return engines_[engine].ready; }
    int running(int engine) const;             // requests it is answering
    size_t size() const { return engines_.size(); }
    const backend_speed & speed(int engine) const { return engines_[engine].speed; }

    struct choice {
        int engine = -1;                // -1: none is ready
        bool load = false;              // one at a time: load this engine in place of the loaded one first
        std::vector<double> seconds;    // the expected time on each engine, loading it first if need be (< 0: can't)
        std::vector<double> if_loaded;  // ...and if it were loaded and idle
        std::vector<double> held;       // the prompt tokens each already holds
        double read = 0;                // the prompt's tokens
        double written = 0;             // the reply's expected tokens
    };
    choice pick(const std::vector<route_piece> & prompt, double max_written, double now_s) const;
    // One at a time: a choice that sent the request to the loaded engine, for the time the others would have saved
    void chosen(const choice & c);

    // A request sent to an engine: that engine holds its prompt from now on. Returns its id for finished().
    int64_t started(int engine, const std::vector<route_piece> & prompt, double expected_s, double now_s);
    // ...and its end: the prompt's bytes and tokens and the tokens written, when the reply says (0 if not)
    void finished(int64_t id, double prompt_bytes, double prompt_tokens, double written_tokens);

    double bytes_per_token() const { return bytes_per_token_; }
    double reply_tokens() const { return reply_tokens_; }

private:
    struct held_piece {
        double end_bytes = 0;  // the prompt's bytes up to the end of this piece
        uint64_t used = 0;     // when it was last sent, in requests
        bool last = false;     // the last piece of a request (what follows it is that engine's own reply)
    };
    struct engine {
        backend_speed speed;
        bool ready = false;
        bool failed = false;
        double load_s = 0;
        double saved_s = 0;   // one at a time: what it would have saved since the last change
        std::unordered_map<uint64_t, held_piece> held;
    };
    struct in_hand {
        int engine;
        double start_s;
        double expected_s;
    };

    double held_bytes(const engine & e, const std::vector<route_piece> & prompt) const;
    double busy_s(int engine, double now_s) const;

    engine_route_options o_;
    std::vector<engine> engines_;
    std::unordered_map<int64_t, in_hand> running_;
    int64_t next_id_ = 1;
    uint64_t clock_ = 0;
    double bytes_per_token_;
    double reply_tokens_;
};

} // namespace kera
