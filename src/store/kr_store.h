/* Weight store (spec P3, section 4): where every weight lives and how it gets to memory.
 *
 * Always-used weights stay resident. Routed experts live in a cache with a learned policy and fall
 * back to the SSD, one read per expert (Phase 2). Frozen in M0; the real store arrives in M5, and
 * tests use the fake in tests/cpp/fake_store.cpp until then.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kr_store kr_store;

/* Called when an expert that wasn't in memory has arrived. ptr stays valid until the store evicts it,
 * which it never does while a read of the same layer is in use. */
typedef void (*kr_ready_cb)(void * user, int layer, int expert, const void * ptr);

/* Returns the expert's bytes if they are in memory. Otherwise returns NULL, starts the read, and calls
 * on_ready when it lands (on_ready may be NULL). */
const void * kr_expert_get(kr_store * s, int layer, int expert, kr_ready_cb on_ready, void * user);

/* Ask for experts that will probably be needed. Reads run in deadline order (section 10.3); a guess
 * with low confidence only uses bandwidth the certain reads leave free. */
void kr_expert_prefetch(kr_store * s, int layer, const int * experts, int n, double deadline_ms, float confidence);

typedef struct {
    uint64_t requests;
    uint64_t hits;           /* already in memory when asked */
    uint64_t misses;         /* had to be read */
    uint64_t prefetched;     /* reads started by kr_expert_prefetch */
    uint64_t prefetch_hits;  /* prefetched experts that were then used */
    double   gb_read;        /* from the SSD */
    double   io_wait_ms;     /* time the caller spent waiting for reads */
} kr_store_stats_t;

void kr_store_stats(kr_store * s, kr_store_stats_t * out);

#ifdef __cplusplus
}
#endif
