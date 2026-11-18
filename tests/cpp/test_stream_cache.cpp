// The expert cache (src/stream/expert_cache.h) against a fake model file: experts land byte for byte where
// the mirror says, the budget holds with least recently used experts going first, an expert's neighbour
// survives its eviction (they share edge pages), early reads count as such, a copy of the file on another
// "drive" serves reads, pins stay, and a layer that needs more than the budget still gets its experts.

#include "kr_test.h"

#include "stream/expert_cache.h"
#include "stream/region.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace kera::stream;

namespace {

uint8_t pattern(uint64_t off) {
    uint64_t x = off * 0x9E3779B97F4A7C15ull;
    x ^= x >> 29;
    return (uint8_t) (x >> 7);
}

struct fake_model {
    std::string path;
    uint64_t size = 0;
    std::vector<layer_layout> layers;
};

// Three MoE layers (model layers 1, 3 and 4) of 8 experts with gate, up and down parts whose sizes aren't
// multiples of the page size, placed at 32-byte (not page) aligned offsets, as in a GGUF file. The parts
// span many pages even at 16 KiB (Apple), so rounding to whole pages changes an expert's size by little and
// the budget tests hold the same number of experts on every OS.
fake_model make_model(const std::string & dir) {
    fake_model m;
    m.path = dir + "/model.gguf";
    uint64_t off = 4096 + 96; // a "header" first
    const uint64_t strides[3] = { 100160, 77760, 123520 };
    for (int layer : { 1, 3, 4 }) {
        layer_layout L;
        L.layer = layer;
        L.n_expert = 8;
        for (int p = 0; p < 3; ++p) {
            L.parts.push_back({ 0, off, strides[p] });
            off += strides[p] * 8 + 32 * (uint64_t) (p + 1); // some padding between tensors
        }
        m.layers.push_back(L);
    }
    m.size = off + 1000;
    std::vector<uint8_t> data(m.size);
    for (uint64_t i = 0; i < m.size; ++i) {
        data[i] = pattern(i);
    }
    std::ofstream(m.path, std::ios::binary).write((const char *) data.data(), (std::streamsize) data.size());
    return m;
}

bool expert_ok(const expert_cache & c, const fake_model & m, int layer, int e) {
    for (const auto & L : m.layers) {
        if (L.layer != layer) {
            continue;
        }
        for (int p = 0; p < (int) L.parts.size(); ++p) {
            const uint8_t * d = c.part_data(layer, e, p);
            const uint64_t start = L.parts[(size_t) p].file_offset + (uint64_t) e * L.parts[(size_t) p].stride;
            for (uint64_t i = 0; i < L.parts[(size_t) p].stride; i += 97) {
                if (d[i] != pattern(start + i)) {
                    return false;
                }
            }
            if (d[L.parts[(size_t) p].stride - 1] != pattern(start + L.parts[(size_t) p].stride - 1)) {
                return false;
            }
        }
    }
    return true;
}

std::string temp_dir(const char * name) {
    const fs::path d = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    fs::create_directories(d);
    return d.u8string();
}

struct setup {
    fake_model m;
    region r;
    expert_cache c;
    uint64_t one = 0; // bytes one expert takes

    bool open(const std::string & dir, uint64_t budget_experts, std::vector<std::vector<std::string>> mirrors = {}, int threads = 4) {
        m = make_model(dir);
        if (!r.reserve(m.size + (1u << 16))) {
            return false;
        }
        cache_options o;
        o.io_threads = threads;
        o.chunk_bytes = 16384;
        o.budget_bytes = 0;
        std::string err;
        // the budget in experts: find one expert's size first
        o.budget_bytes = 1ull << 40;
        if (!c.open({ m.path }, { 0 }, mirrors, &r, m.layers, o, &err)) {
            std::printf("  open failed: %s\n", err.c_str());
            return false;
        }
        one = c.expert_bytes(1);
        c.close();
        o.budget_bytes = budget_experts * one;
        return c.open({ m.path }, { 0 }, mirrors, &r, m.layers, o, &err);
    }
};

} // namespace

KR_TEST(experts_land_byte_for_byte) {
    setup s;
    KR_CHECK(s.open(temp_dir("kera_stream_a"), 100));
    const int32_t ids[] = { 0, 3, 3, 7 };
    KR_CHECK(s.c.require(1, ids, 4));
    KR_CHECK(expert_ok(s.c, s.m, 1, 0) && expert_ok(s.c, s.m, 1, 3) && expert_ok(s.c, s.m, 1, 7));
    KR_CHECK(s.c.is_resident(1, 3) && !s.c.is_resident(1, 4));
    const int32_t more[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    KR_CHECK(s.c.require(4, more, 8));
    for (int e = 0; e < 8; ++e) {
        KR_CHECK(expert_ok(s.c, s.m, 4, e));
    }
    const auto st = s.c.stats();
    KR_CHECK(st.requests == 11 && st.misses == 11 && st.hits == 0);
    KR_CHECK(st.bytes_read >= 11 * (10016 + 7776 + 12352));
    KR_CHECK(s.c.require(1, ids, 4));
    KR_CHECK(s.c.stats().hits == 3);
}

KR_TEST(budget_holds_least_recently_used_first) {
    setup s;
    KR_CHECK(s.open(temp_dir("kera_stream_b"), 4)); // room for 4 experts
    for (int e = 0; e < 4; ++e) {
        const int32_t id = e;
        KR_CHECK(s.c.require(3, &id, 1));
    }
    KR_CHECK(s.c.stats().used_bytes <= 4 * s.one);
    const int32_t again = 0; // expert 0 becomes the most recently used
    KR_CHECK(s.c.require(3, &again, 1));
    const int32_t fresh = 5; // evicts expert 1, the least recently used
    KR_CHECK(s.c.require(3, &fresh, 1));
    KR_CHECK(s.c.is_resident(3, 0) && !s.c.is_resident(3, 1) && s.c.is_resident(3, 2) && s.c.is_resident(3, 5));
    KR_CHECK(s.c.stats().evictions == 1);
    KR_CHECK(s.c.stats().used_bytes <= 4 * s.one);
    KR_CHECK(expert_ok(s.c, s.m, 3, 0) && expert_ok(s.c, s.m, 3, 5));
}

KR_TEST(a_neighbour_survives_eviction) {
    // experts 2 and 3 of a part share a page; evicting 2 must leave 3 intact
    setup s;
    KR_CHECK(s.open(temp_dir("kera_stream_c"), 2));
    const int32_t pair[] = { 2, 3 };
    KR_CHECK(s.c.require(1, pair, 2));
    const int32_t keep = 3; // 3 is now the most recently used
    KR_CHECK(s.c.require(1, &keep, 1));
    const int32_t other = 6; // evicts 2
    KR_CHECK(s.c.require(1, &other, 1));
    KR_CHECK(!s.c.is_resident(1, 2) && s.c.is_resident(1, 3));
    KR_CHECK(expert_ok(s.c, s.m, 1, 3) && expert_ok(s.c, s.m, 1, 6));
    const int32_t back = 2; // and 2 comes back right
    KR_CHECK(s.c.require(1, &back, 1));
    KR_CHECK(expert_ok(s.c, s.m, 1, 2));
}

KR_TEST(a_read_never_writes_a_neighbours_bytes) {
    // expert 3's first page of each part also holds the end of expert 2's: reading 3 must leave those bytes alone, so
    // two reads of one page never meet (N-012: on an encrypted drive such a page could end up wrong)
    setup s;
    KR_CHECK(s.open(temp_dir("kera_stream_g"), 100));
    const int32_t three = 3;
    KR_CHECK(s.c.require(1, &three, 1));
    KR_CHECK(expert_ok(s.c, s.m, 1, 3));
    const uintptr_t page = region::page_size();
    for (int p = 0; p < 3; ++p) {
        const uint8_t * start = s.c.part_data(1, 3, p);
        const uint8_t * first_page = (const uint8_t *) ((uintptr_t) start / page * page);
        KR_CHECK(first_page < start); // the test model's parts don't start on a page
        size_t written = 0;
        for (const uint8_t * q = first_page; q < start; ++q) {
            written += *q != 0;
        }
        KR_CHECK(written == 0);
    }
    const int32_t two = 2; // and 2 then arrives right next to 3
    KR_CHECK(s.c.require(1, &two, 1));
    KR_CHECK(expert_ok(s.c, s.m, 1, 2) && expert_ok(s.c, s.m, 1, 3));
    const int32_t both[] = { 4, 5 }; // two neighbours read at once
    KR_CHECK(s.c.require(1, both, 2));
    KR_CHECK(expert_ok(s.c, s.m, 1, 4) && expert_ok(s.c, s.m, 1, 5));
}

KR_TEST(early_reads_and_a_layer_bigger_than_the_budget) {
    setup s;
    KR_CHECK(s.open(temp_dir("kera_stream_d"), 3));
    const int32_t guess[] = { 1, 2 };
    s.c.prefetch(4, guess, 2, expert_cache::reason::lookahead);
    const int32_t first = 0;
    KR_CHECK(s.c.require(3, &first, 1)); // another layer runs while the guesses load
    const int32_t used[] = { 1, 2 };
    KR_CHECK(s.c.require(4, used, 2));
    auto st = s.c.stats();
    KR_CHECK(st.prefetched == 2 && st.prefetch_used == 2 && st.hits + st.late == 2);
    KR_CHECK(expert_ok(s.c, s.m, 4, 1) && expert_ok(s.c, s.m, 4, 2));
    // six experts at once with room for three: all six arrive, then the cache shrinks back
    const int32_t six[] = { 0, 1, 2, 3, 4, 5 };
    KR_CHECK(s.c.require(1, six, 6));
    for (int e = 0; e < 6; ++e) {
        KR_CHECK(expert_ok(s.c, s.m, 1, e));
    }
    st = s.c.stats();
    KR_CHECK(st.overflow_steps == 1);
    const int32_t one = 7;
    KR_CHECK(s.c.require(3, &one, 1));
    KR_CHECK(s.c.stats().used_bytes <= 3 * s.one);
}

KR_TEST(a_copy_on_another_drive_and_pins) {
    const std::string dir = temp_dir("kera_stream_e");
    const std::string copy_dir = temp_dir("kera_stream_e2");
    setup s;
    fake_model probe = make_model(copy_dir); // the same bytes in another place
    KR_CHECK(s.open(dir, 4, { { probe.path } }, 4));
    const int32_t all[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const int32_t pinned = 7;
    s.c.pin(3, &pinned, 1, false);
    for (int layer : { 1, 3, 4 }) {
        for (int e = 0; e < 8; ++e) {
            KR_CHECK(s.c.require(layer, &all[e], 1));
            KR_CHECK(expert_ok(s.c, s.m, layer, e));
        }
    }
    KR_CHECK(s.c.is_resident(3, 7)); // pinned: never evicted
    const uint64_t p1 = s.c.stats().pinned_bytes;
    KR_CHECK(p1 > 0 && p1 <= s.one);
    // prompt pins come off again
    const int32_t pp = 0;
    s.c.pin(4, &pp, 1, true);
    KR_CHECK(s.c.stats().pinned_bytes > p1);
    s.c.unpin_prompt();
    KR_CHECK(s.c.stats().pinned_bytes == p1);
}

KR_TEST(usage_counts_each_layer_evaluation_once) {
    setup s;
    KR_CHECK(s.open(temp_dir("kera_stream_f"), 100));
    const int32_t a[] = { 1, 1, 1, 2 };
    KR_CHECK(s.c.require(1, a, 4));
    KR_CHECK(s.c.require(1, a, 4));
    const auto u = s.c.usage(1);
    KR_CHECK(u.size() == 8 && u[1] == 2 && u[2] == 2 && u[0] == 0);
}

KR_TEST_MAIN()
