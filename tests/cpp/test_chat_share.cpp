// How several chats share the chip (src/planner/chat_share.h), on a simulated chip shaped like the Minisforum deep in
// a long document: a step that only writes takes 30 ms, and each prompt token 5 ms more (a 2,048-token step: 10 s).

#include "kr_test.h"

#include "planner/chat_share.h"

#include <algorithm>

using namespace kera;

namespace {

const double k_write_step = 0.030;
const double k_per_token = 0.005;
const int32_t k_batch = 2048;

struct run {
    double seconds = 0;
    double reading = 0;   // seconds of steps that read prompt tokens
    double longest = 0;   // the longest step while someone wrote
    int words = 0;        // words the writers got (one per step each)
};

// one long prompt read while n_writing chats write, until `seconds` have passed or the prompt is read
run simulate(chat_share & s, int32_t n_writing, int32_t prompt, double seconds) {
    run r;
    int32_t left = prompt;
    while (r.seconds < seconds && left > 0) {
        std::vector<chat_share::waiting> w = { { 0, left, 1 } };
        const int32_t n = std::min(left, s.plan(w, n_writing, k_batch));
        const double t = k_write_step + n * k_per_token;
        s.step(n, n_writing, t);
        left -= n;
        r.seconds += t;
        r.reading += n > 0 ? t : 0;
        r.words += n_writing;
        if (n_writing > 0) {
            r.longest = std::max(r.longest, t);
        }
    }
    return r;
}

} // namespace

KR_TEST(with_nobody_writing_prompts_get_whole_batches) {
    chat_share s(chat_share_options{});
    std::vector<chat_share::waiting> w = { { 0, 50000, 1 } };
    KR_CHECK(s.plan(w, 0, k_batch) == k_batch);
    const run r = simulate(s, 0, 20000, 1e9);
    KR_CHECK(r.reading == r.seconds); // every step read the prompt
}

KR_TEST(writers_keep_writing_while_a_long_document_is_read) {
    chat_share s(chat_share_options{});
    const run r = simulate(s, 2, 1000000, 120);
    // about half the time goes to the document, in steps of about half a second
    const double share = r.reading / r.seconds;
    KR_CHECK(share > 0.4 && share < 0.6);
    KR_CHECK(r.longest < 0.75);
    // the writers get about half their speed alone (33 words a second each), not one word per 10-second step
    const double words_per_s = r.words / 2.0 / r.seconds;
    KR_CHECK(words_per_s > 12 && words_per_s < 22);
}

KR_TEST(the_read_share_can_be_turned_off) {
    chat_share_options o;
    o.read_share = 0;
    chat_share s(o);
    std::vector<chat_share::waiting> w = { { 0, 50000, 1 } };
    KR_CHECK(s.plan(w, 3, k_batch) == k_batch);
}

KR_TEST(a_new_message_goes_first_without_waiting_for_the_writers_time) {
    chat_share s(chat_share_options{});
    // a long document's last step spent the time prompts were allowed: writers are owed theirs
    int guard = 0;
    while (s.bank_s() > 0 && guard++ < 1000) {
        std::vector<chat_share::waiting> doc = { { 0, 1000000, 1 } };
        const int32_t n = s.plan(doc, 1, k_batch);
        s.step(n, 1, k_write_step + n * k_per_token);
    }
    KR_CHECK(s.bank_s() <= 0);
    std::vector<chat_share::waiting> doc = { { 0, 40000, 1 } };
    KR_CHECK(s.plan(doc, 1, k_batch) == 0); // the document waits
    std::vector<chat_share::waiting> w = { { 0, 40000, 1 }, { 1, 300, 2 }, { 2, 9000, 3 }, { 3, 120, 4 } };
    const int32_t budget = s.plan(w, 1, k_batch);
    KR_CHECK(w[0].slot == 3 && w[1].slot == 1 && w[2].slot == 2 && w[3].slot == 0); // fewest tokens left first
    // the new messages are read now, in steps of about half a second (0.5 s at 5 ms a token)
    KR_CHECK(budget >= 80 && budget <= 120);
}

KR_TEST(a_lower_priority_goes_first) {
    chat_share s(chat_share_options{});
    std::vector<chat_share::waiting> w = { { 0, 100, 1, 0 }, { 1, 5000, 2, -1 }, { 2, 50, 3, 0 } };
    s.plan(w, 0, k_batch);
    KR_CHECK(w[0].slot == 1 && w[1].slot == 2 && w[2].slot == 0); // priority, then the fewest tokens left
    chat_share_options o;
    o.short_first = false;
    chat_share t(o);
    std::vector<chat_share::waiting> v = { { 0, 100, 1, 0 }, { 1, 5000, 2, -1 }, { 2, 50, 3, 0 } };
    t.plan(v, 0, k_batch);
    KR_CHECK(v[0].slot == 1 && v[1].slot == 0 && v[2].slot == 2); // priority, then as they were
}

KR_TEST(without_short_first_the_order_stays) {
    chat_share_options o;
    o.short_first = false;
    chat_share s(o);
    std::vector<chat_share::waiting> w = { { 0, 40000, 1 }, { 1, 300, 2 } };
    s.plan(w, 0, k_batch);
    KR_CHECK(w[0].slot == 0 && w[1].slot == 1);
}

KR_TEST(the_step_size_follows_the_measured_cost) {
    chat_share s(chat_share_options{});
    simulate(s, 1, 1000000, 20);
    // 0.5 s at 5 ms a token: about 100 tokens a step
    KR_CHECK(s.seconds_per_token() > 0.004 && s.seconds_per_token() < 0.006);
    KR_CHECK(s.writing_step_s() > 0.025 && s.writing_step_s() < 0.035);
}

KR_TEST_MAIN()
