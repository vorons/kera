// How several chats share the chip (the owner's request of 17 Oct, docs/DECISIONS.md D-023; llama.cpp hook 7).
// Upstream's server reads waiting prompts in slot order, each as far as a batch allows, so one person's long document
// holds up everyone else's new messages, and every step of it holds up their writing: deep in a long document on the
// Minisforum, one 2,048-token step takes several seconds, and the others write one word per step. Here the prompt with
// the fewest tokens left (a new message) is read first, and while others write, prompt reading goes in steps of about
// half a second and a long one takes at most a share of the chip's time, so they keep writing. With nobody writing,
// prompts get whole batches, as upstream.
#pragma once

#include <cstdint>
#include <vector>

namespace kera {

struct chat_share_options {
    bool short_first = true;  // the prompt with the fewest tokens left is read first (after priority)
    double read_share = 0.5;  // while others write: the most of the chip's time prompt reading takes (0 = no limit)
    double step_s = 0.5;      // while others write: about the longest one step of prompt reading takes
};

class chat_share {
public:
    struct waiting {
        int32_t slot;          // the server's slot
        int32_t left;          // prompt tokens still to read
        int64_t since;         // arrival order
        int32_t priority = 0;  // the request's: lower is sooner (vLLM's "priority")
    };

    explicit chat_share(const chat_share_options & o) : o_(o), bank_s_(o.step_s) {}

    // Puts the waiting prompts in the order to read them this step, and returns the most prompt tokens the step may
    // read, with n_writing chats writing in it (n_batch: the most one batch holds)
    int32_t plan(std::vector<waiting> & w, int32_t n_writing, int32_t n_batch) const;

    // A finished step: n_prompt prompt tokens read and n_writing chats written for, in `seconds`
    void step(int32_t n_prompt, int32_t n_writing, double seconds);

    double seconds_per_token() const { return t_tok_; }  // reading one prompt token, lately (0 = not measured yet)
    double writing_step_s() const { return t_dec_; }      // a step that only writes, lately
    double bank_s() const { return bank_s_; }             // prompt time allowed before writers are owed theirs

private:
    int32_t chunk(int32_t n_batch) const; // the prompt tokens that take about step_s

    chat_share_options o_;
    double t_dec_ = 0;
    double t_tok_ = 0;
    double bank_s_ = 0;
};

} // namespace kera
