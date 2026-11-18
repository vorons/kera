// Which graphics back-end serves a model faster, when Kera has two for one chip (Vulkan and the optional ROCm add-on
// on AMD; the owner's request of 17 Oct, docs/DECISIONS.md D-022). Each back-end is measured once on the model
// (kera calibrate): the time per prompt token and per written token, both growing with how much the chat already
// holds. The pick is the back-end with the shorter expected time for the replies the model is asked for: the ones
// it served before, or, with too few of those, chats plus now and then a long document.
#pragma once

#include <string>
#include <vector>

namespace kera {

// One back-end on one model. Reading n prompt tokens after d already held takes about
// n * (read_s + read_s_per_depth * (d + n / 2)); writing g tokens after d held, g * (write_s + write_s_per_depth * (d + g / 2)).
struct backend_speed {
    std::string backend;          // "vulkan", "hip"
    double read_s = 0;            // seconds per prompt token with nothing held
    double read_s_per_depth = 0;  // more seconds per prompt token for each token held
    double write_s = 0;           // seconds per written token with nothing held
    double write_s_per_depth = 0;

    bool ok() const { return read_s > 0 && write_s > 0; }
    double read_seconds(double held, double n) const;
    double write_seconds(double held, double g) const;
    double reply_seconds(double held, double read, double written) const;
};

// Fits a back-end's speeds from two prompt reads (n tokens after d held, taking t seconds) and two stretches of
// writing (g tokens after d held). The slopes are never negative: noise that makes one so is read as no slope.
backend_speed fit_backend_speed(const std::string & backend,
                                double read_d1, double read_n1, double read_t1,
                                double read_d2, double read_n2, double read_t2,
                                double write_d1, double write_g1, double write_t1,
                                double write_d2, double write_g2, double write_t2);

struct reply_shape {
    double held = 0;     // tokens already in memory
    double read = 0;     // prompt tokens read
    double written = 0;  // tokens written
    double weight = 1;
};

// The replies to weigh for a model with too little history: chat messages (a short prompt after a short chat), and
// one in ten a document half the context long (at most 64K tokens).
std::vector<reply_shape> default_replies(int n_ctx);

struct backend_pick {
    std::string backend;     // the faster one ("" when none was measured)
    double seconds = 0;      // its expected time per reply
    std::string other;       // the next best ("" if only one was measured)
    double other_seconds = 0;
};

backend_pick pick_backend(const std::vector<backend_speed> & measured, const std::vector<reply_shape> & replies);

} // namespace kera
