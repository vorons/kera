// A fake weight store behind the frozen kr_store interface (spec P3: each part is tested against fakes
// of the others). Experts are "read" from a backing array into an LRU of a few slots; the test checks
// hits, misses, prefetch accounting and that the bytes handed out are the right expert's.

#include "kr_test.h"

#include "store/kr_store.h"

#include <cstring>
#include <list>
#include <map>
#include <utility>
#include <vector>

struct kr_store {
    int n_layers;
    int n_experts;
    size_t expert_bytes;
    size_t slots;
    std::vector<uint8_t> backing;                       // [layer][expert][bytes]
    std::list<std::pair<int, int>> lru;                 // front = most recently used
    std::map<std::pair<int, int>, std::vector<uint8_t>> cache;
    std::map<std::pair<int, int>, bool> prefetched;
    kr_store_stats_t stats{};

    const uint8_t * src(int l, int e) const {
        return backing.data() + ((size_t) l * n_experts + e) * expert_bytes;
    }

    void touch(std::pair<int, int> key) {
        lru.remove(key);
        lru.push_front(key);
    }

    const uint8_t * load(std::pair<int, int> key) {
        if (cache.size() >= slots) {
            const auto victim = lru.back();
            lru.pop_back();
            cache.erase(victim);
            prefetched.erase(victim);
        }
        auto & buf = cache[key];
        buf.assign(src(key.first, key.second), src(key.first, key.second) + expert_bytes);
        stats.gb_read += expert_bytes / 1e9;
        touch(key);
        return buf.data();
    }
};

extern "C" const void * kr_expert_get(kr_store * s, int layer, int expert, kr_ready_cb on_ready, void * user) {
    s->stats.requests++;
    const auto key = std::make_pair(layer, expert);
    const auto it = s->cache.find(key);
    if (it != s->cache.end()) {
        s->stats.hits++;
        if (s->prefetched[key]) {
            s->stats.prefetch_hits++;
            s->prefetched[key] = false;
        }
        s->touch(key);
        return it->second.data();
    }
    s->stats.misses++;
    const uint8_t * p = s->load(key); // the fake "reads" synchronously, then reports arrival
    if (on_ready) {
        on_ready(user, layer, expert, p);
    }
    return nullptr;
}

extern "C" void kr_expert_prefetch(kr_store * s, int layer, const int * experts, int n, double, float) {
    for (int i = 0; i < n; ++i) {
        const auto key = std::make_pair(layer, experts[i]);
        if (!s->cache.count(key)) {
            s->load(key);
            s->prefetched[key] = true;
            s->stats.prefetched++;
        }
    }
}

extern "C" void kr_store_stats(kr_store * s, kr_store_stats_t * out) {
    *out = s->stats;
}

static kr_store make_store() {
    kr_store s;
    s.n_layers = 2;
    s.n_experts = 8;
    s.expert_bytes = 256;
    s.slots = 4;
    s.backing.resize((size_t) s.n_layers * s.n_experts * s.expert_bytes);
    for (size_t i = 0; i < s.backing.size(); ++i) {
        s.backing[i] = (uint8_t) (i / s.expert_bytes); // every byte of expert (l, e) is l*8 + e
    }
    return s;
}

KR_TEST(miss_then_hit_and_right_bytes) {
    kr_store s = make_store();
    int arrived = 0;
    const void * p = kr_expert_get(&s, 1, 3, [](void * u, int, int, const void * ptr) {
        (*(int *) u)++;
        KR_CHECK(((const uint8_t *) ptr)[0] == 1 * 8 + 3);
    }, &arrived);
    KR_CHECK(p == nullptr);
    KR_CHECK(arrived == 1);
    p = kr_expert_get(&s, 1, 3, nullptr, nullptr);
    KR_CHECK(p != nullptr && ((const uint8_t *) p)[255] == 11);
    kr_store_stats_t st;
    kr_store_stats(&s, &st);
    KR_CHECK(st.requests == 2 && st.hits == 1 && st.misses == 1);
}

KR_TEST(prefetch_turns_misses_into_hits) {
    kr_store s = make_store();
    const int guess[] = { 0, 1, 2 };
    kr_expert_prefetch(&s, 0, guess, 3, 5.0, 0.8f);
    for (int e : guess) {
        KR_CHECK(kr_expert_get(&s, 0, e, nullptr, nullptr) != nullptr);
    }
    kr_store_stats_t st;
    kr_store_stats(&s, &st);
    KR_CHECK(st.prefetched == 3 && st.prefetch_hits == 3 && st.misses == 0);
}

KR_TEST(lru_evicts_the_oldest) {
    kr_store s = make_store();
    for (int e = 0; e < 5; ++e) {
        kr_expert_get(&s, 0, e, nullptr, nullptr); // 5 experts, 4 slots: expert 0 is evicted
    }
    KR_CHECK(kr_expert_get(&s, 0, 4, nullptr, nullptr) != nullptr);
    KR_CHECK(kr_expert_get(&s, 0, 0, nullptr, nullptr) == nullptr);
}

KR_TEST_MAIN()
