// How several chats share the chip (chat_share.h).

#include "planner/chat_share.h"

#include <algorithm>

namespace kera {

int32_t chat_share::chunk(int32_t n_batch) const {
    if (t_tok_ <= 0) {
        return std::min(n_batch, 64); // not measured yet: a short first step, which measures the cost
    }
    // clamped as a double first: after many quick prompt steps t_tok_ is tiny and the quotient overflows an int
    const double n = std::min<double>(o_.step_s / t_tok_, (double) n_batch);
    return std::clamp((int32_t) n, std::min(n_batch, 16), n_batch);
}

int32_t chat_share::plan(std::vector<waiting> & w, int32_t n_writing, int32_t n_batch) const {
    // a lower priority first; then the prompt with the fewest tokens left (the shortest wait on average); then the one
    // that came first
    const bool short_first = o_.short_first;
    std::stable_sort(w.begin(), w.end(), [short_first](const waiting & a, const waiting & b) {
        if (a.priority != b.priority) {
            return a.priority < b.priority;
        }
        if (short_first && a.left != b.left) {
            return a.left < b.left;
        }
        return short_first && a.since < b.since;
    });
    if (n_writing == 0 || o_.read_share <= 0 || w.empty()) {
        return n_batch; // nobody else is writing: whole batches, as upstream
    }
    const int32_t c = chunk(n_batch);
    // a short prompt (one batch's worth) doesn't wait for the time writers are owed: it is someone's new message,
    // and their answer starts once it is read. It still goes in steps of about step_s, so writers keep writing
    if (o_.short_first && w.front().left <= n_batch) {
        return c;
    }
    return bank_s_ > 0 ? c : 0;
}

void chat_share::step(int32_t n_prompt, int32_t n_writing, double seconds) {
    if (seconds <= 0) {
        return;
    }
    double reading = 0;
    if (n_prompt == 0) {
        if (n_writing > 0) {
            t_dec_ = t_dec_ > 0 ? 0.8 * t_dec_ + 0.2 * seconds : seconds;
        }
    } else {
        reading = std::max(0.0, seconds - (n_writing > 0 ? t_dec_ : 0.0));
        const double per = reading / n_prompt;
        t_tok_ = t_tok_ > 0 ? 0.7 * t_tok_ + 0.3 * per : per;
    }
    if (n_writing == 0 || o_.read_share <= 0) {
        bank_s_ = o_.step_s; // nobody to share with: a full allowance for when somebody starts writing
        return;
    }
    // every step earns prompts their share of its time, and a prompt step spends what its reading took
    bank_s_ = std::clamp(bank_s_ + o_.read_share * seconds - reading, -o_.step_s, o_.step_s);
}

} // namespace kera
