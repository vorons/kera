#pragma once

// Kera hook 6 (listed in Kera's docs/UPSTREAM_PATCHES.md): an optional observer told the shape of every reply the
// server finishes, so Kera can learn how a model is used (how much of each prompt was already in memory, how much
// was read, how much written) and pick the back-end that serves that fastest. Without one set, the server behaves
// exactly as upstream.

#include <cstdint>

struct common_reply_observer {
    virtual ~common_reply_observer() = default;

    // A reply finished: n_held tokens of its prompt were already in memory, n_read were read in t_read_ms, and
    // n_written were written in t_write_ms. Called on the server's thread: keep it short.
    virtual void reply(int32_t n_held, int32_t n_read, int32_t n_written, double t_read_ms, double t_write_ms) = 0;
};

// Set before the server starts; nullptr (the default) observes nothing.
void                    common_set_reply_observer(common_reply_observer * obs);
common_reply_observer * common_get_reply_observer();
