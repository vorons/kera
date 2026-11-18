// The back-end choice (src/planner/backend_pick.h) with the Minisforum's measured speeds on Nemotron 3 Nano (16 Oct,
// docs/BENCHMARKS.md): Vulkan reads a short prompt faster and writes faster, but its reading slows ten times as fast
// with how much the chat holds, so ROCm wins once documents are long. The fit must recover what it was fed.

#include "kr_test.h"

#include "planner/backend_pick.h"

#include <cmath>

using namespace kera;

namespace {

// the 16 Oct measurements (a 7.7K and a 95K-token document read from empty; writing after them)
backend_speed vulkan() {
    return fit_backend_speed("vulkan", 0, 7694, 6.90, 0, 95412, 295.93, 7694, 300, 300 / 57.3, 95412, 300, 300 / 43.1);
}

backend_speed rocm() {
    return fit_backend_speed("hip", 0, 7694, 9.11, 0, 95412, 138.20, 7694, 300, 300 / 45.4, 95412, 300, 300 / 38.8);
}

bool near(double a, double b, double rel) {
    return std::fabs(a - b) <= rel * std::fabs(b);
}

} // namespace

KR_TEST(the_fit_gives_back_what_it_was_fed) {
    const backend_speed v = vulkan();
    KR_CHECK(v.ok());
    KR_CHECK(near(v.read_seconds(0, 7694), 6.90, 0.01) && near(v.read_seconds(0, 95412), 295.93, 0.01));
    KR_CHECK(near(v.write_seconds(7694, 300), 300 / 57.3, 0.01));
    // and between the points, the 64K document took 147.9 s
    KR_CHECK(near(v.read_seconds(0, 64016), 147.94, 0.05));
    const backend_speed r = rocm();
    KR_CHECK(near(r.read_seconds(0, 64016), 84.95, 0.05));
}

KR_TEST(chats_go_to_vulkan_long_documents_to_rocm) {
    const std::vector<backend_speed> both = { vulkan(), rocm() };
    const backend_pick chat = pick_backend(both, { { 1500, 200, 300, 1 } });
    KR_CHECK(chat.backend == "vulkan" && chat.other == "hip" && chat.seconds < chat.other_seconds);
    const backend_pick doc = pick_backend(both, { { 0, 64000, 300, 1 } });
    KR_CHECK(doc.backend == "hip");
    // the crossover is a prompt of about 20K tokens read from empty
    KR_CHECK(pick_backend(both, { { 0, 12000, 0, 1 } }).backend == "vulkan");
    KR_CHECK(pick_backend(both, { { 0, 30000, 0, 1 } }).backend == "hip");
}

KR_TEST(without_history_the_context_size_decides) {
    const std::vector<backend_speed> both = { vulkan(), rocm() };
    KR_CHECK(pick_backend(both, default_replies(16384)).backend == "vulkan");   // documents of 8K at most
    KR_CHECK(pick_backend(both, default_replies(131072)).backend == "hip");     // documents of 64K
}

KR_TEST(one_measurement_or_none) {
    backend_speed broken;
    broken.backend = "hip";
    const backend_pick one = pick_backend({ vulkan(), broken }, default_replies(131072));
    KR_CHECK(one.backend == "vulkan" && one.other.empty());
    KR_CHECK(pick_backend({}, default_replies(4096)).backend.empty());
}

KR_TEST(noise_never_makes_a_slope_negative) {
    // the deeper read measured a little faster than the shallow one
    const backend_speed s = fit_backend_speed("x", 0, 2048, 2.0, 2048, 14336, 13.9, 512, 32, 0.6, 16384, 32, 0.59);
    KR_CHECK(s.ok() && s.read_s_per_depth == 0 && s.write_s_per_depth == 0);
}

KR_TEST_MAIN()
