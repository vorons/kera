// llama.cpp's scheduler (a fix in third_party, docs/UPSTREAM_PATCHES.md): an eval callback that returns false
// cancels the whole graph compute, as ggml-backend.h says. Before, it only left the current split and computed the
// rest from data that was never written, so a failed read of streamed experts gave garbage instead of an error
// (docs/NEGATIVE_RESULTS.md N-012).

#include "kr_test.h"

#include "ggml-backend.h"
#include "ggml.h"

#include <cstring>

namespace {

struct cb_state {
    int called = 0;
    bool stop = true;
};

bool stop_at_b(ggml_tensor * t, bool ask, void * user) {
    auto * st = (cb_state *) user;
    if (ask) {
        return std::strcmp(t->name, "b") == 0;
    }
    st->called++;
    return !st->stop;
}

ggml_status run(bool stop, float * out, int * called) {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        return GGML_STATUS_FAILED;
    }
    ggml_backend_t backends[] = { cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 1, 64, false, false);
    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4);
    ggml_set_name(a, "a");
    ggml_set_input(a);
    ggml_tensor * b = ggml_scale(ctx, a, 2.0f);
    ggml_set_name(b, "b");
    ggml_tensor * c = ggml_scale(ctx, b, 3.0f);
    ggml_set_name(c, "c");
    ggml_set_output(c);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, c);

    cb_state st;
    st.stop = stop;
    ggml_backend_sched_set_eval_callback(sched, stop_at_b, &st);
    ggml_status status = GGML_STATUS_ALLOC_FAILED;
    if (ggml_backend_sched_alloc_graph(sched, gf)) {
        const float in[4] = { 1, 2, 3, 4 };
        ggml_backend_tensor_set(a, in, 0, sizeof(in));
        status = ggml_backend_sched_graph_compute(sched, gf);
        if (status == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(c, out, 0, 4 * sizeof(float));
        }
    }
    *called = st.called;
    ggml_free(ctx);
    ggml_backend_sched_free(sched);
    ggml_backend_free(cpu);
    return status;
}

} // namespace

KR_TEST(a_callback_that_returns_true_lets_the_graph_finish) {
    float out[4] = {};
    int called = 0;
    KR_CHECK(run(false, out, &called) == GGML_STATUS_SUCCESS);
    KR_CHECK(called == 1);
    KR_CHECK_NEAR(out[3], 24.0, 1e-6);
}

KR_TEST(a_callback_that_returns_false_cancels_the_graph) {
    float out[4] = {};
    int called = 0;
    KR_CHECK(run(true, out, &called) == GGML_STATUS_ABORTED);
    KR_CHECK(called == 1);
}

KR_TEST_MAIN()
