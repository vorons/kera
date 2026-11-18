#pragma once

// Kera hook 7 (listed in Kera's docs/UPSTREAM_PATCHES.md): an optional scheduler for when several chats share the
// server. Upstream reads waiting prompts in slot order, each as far as the batch allows, so one person's long document
// holds up everyone else's new messages, and every step of it holds up their writing. A scheduler sets the order
// waiting prompts are read in and how many prompt tokens a step may take while other chats write, and can let a
// prompt share the start it has in common with another chat's memory instead of reading it again. With one set, a
// request's "priority" (lower is sooner, as vLLM's) also orders the requests waiting for a free slot. Without one set,
// the server behaves exactly as upstream.

#include <cstdint>
#include <vector>

struct common_batch_scheduler {
    virtual ~common_batch_scheduler() = default;

    struct prompt {
        int32_t slot;     // the slot's id
        int32_t left;     // prompt tokens still to read (for a prompt not started yet, beyond what its slot holds)
        int64_t since;    // when its request came (the task's id: lower is earlier)
        int32_t priority; // the request's priority: lower is sooner (0 unless it sets one)
    };

    // This step: reorders `prompts` into the order to read them in, and returns the most prompt tokens the step may
    // read, with n_writing chats writing in it (n_batch: the most a batch holds)
    virtual int32_t plan(std::vector<prompt> & prompts, int32_t n_writing, int32_t n_batch) = 0;

    // A step finished: it read n_prompt prompt tokens and wrote for n_writing chats, in ms milliseconds
    virtual void step(int32_t n_prompt, int32_t n_writing, double ms) = 0;

    // Whether a new prompt may share the start it has in common with another chat's memory, and from how many
    // tokens on. The server only does it where the memory of a position never changes once read: plain attention
    // memory, one memory for all chats, no draft model, no context shift
    virtual int32_t share_min_tokens() const = 0; // 0 = never
};

// Set before the server starts; nullptr (the default) keeps upstream's behaviour.
void                     common_set_batch_scheduler(common_batch_scheduler * s);
common_batch_scheduler * common_get_batch_scheduler();
