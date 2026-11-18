// Saved chat states on disk: the file round-trips and its checksum catches damage; the store finds the
// longest shared start, replaces what a newer save covers, evicts within its budget, and keeps its
// entries across restarts.

#include "kr_test.h"

#include "kvstore/kr_prompt_disk.h"

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static std::string temp_dir(const char * name) {
    const fs::path d = fs::temp_directory_path() / name;
    std::error_code ec;
    fs::remove_all(d, ec);
    return d.u8string();
}

static kera::saved_prompt_blobs make(std::vector<int32_t> tokens, size_t blob_bytes, uint8_t fill) {
    kera::saved_prompt_blobs p;
    p.tokens = std::move(tokens);
    p.meta = { 1, -1, 42 };
    p.blobs = { std::vector<uint8_t>(blob_bytes, fill), std::vector<uint8_t>(3, (uint8_t) (fill + 1)), {} };
    return p;
}

static std::vector<int32_t> seq(int n, int start = 0) {
    std::vector<int32_t> v((size_t) n);
    for (int i = 0; i < n; ++i) {
        v[(size_t) i] = start + i;
    }
    return v;
}

KR_TEST(file_round_trip_and_checksum) {
    const std::string dir = temp_dir("kera_test_kvp");
    fs::create_directories(dir);
    const std::string path = dir + "/a.kvp";
    const auto p = make(seq(100), 5000, 7);
    KR_CHECK(kera::write_saved_prompt(path, p));
    kera::saved_prompt_blobs q;
    KR_CHECK(kera::read_saved_prompt(path, q));
    KR_CHECK(q.tokens == p.tokens && q.meta == p.meta && q.blobs == p.blobs);
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(600);
        f.put((char) 0x55); // damage a byte inside the first blob
    }
    KR_CHECK(!kera::read_saved_prompt(path, q));
    KR_CHECK(kera::read_saved_prompt(path, q, true)); // the tokens alone are still readable
    fs::remove_all(dir);
}

KR_TEST(longest_start_wins_and_covered_entries_go) {
    const std::string dir = temp_dir("kera_test_store");
    {
        kera::prompt_disk_store s(dir, 1ull << 30);
        s.save_async(make(seq(50), 100, 1));          // chat A, turn 1
        s.flush();
        s.save_async(make(seq(80), 100, 2));          // chat A, turn 2: covers turn 1
        s.save_async(make(seq(60, 1000), 100, 3));    // chat B
        s.flush();
        KR_CHECK(s.entries() == 2);
        auto turn3 = seq(78);                         // chat A again, its last answer written back differently
        turn3.push_back(-5);
        turn3.push_back(-6);
        turn3.push_back(-7);
        s.save_async(make(turn3, 100, 2));
        s.flush();
        KR_CHECK(s.entries() == 2);                   // 78 of turn 2's 80 tokens shared: turn 2 goes
        s.save_async(make(seq(80), 100, 2));          // (put turn 2 back for the checks below)
        s.flush();
        kera::saved_prompt_blobs out;
        size_t lcp = 0;
        auto q = seq(80);
        q.push_back(9999);                            // chat A, turn 3
        KR_CHECK(s.load_best(q, 0, 0.25, out, &lcp));
        KR_CHECK(lcp == 80 && out.blobs[0][0] == 2);
        KR_CHECK(!s.load_best(q, 80, 0.25, out));     // nothing longer than what RAM already has
        KR_CHECK(!s.load_best(seq(10), 0, 0.25, out)); // 10 of 80 tokens is under a quarter: not worth it
        KR_CHECK(!s.load_best(seq(5, 5000), 0, 0.25, out));
    }
    {
        kera::prompt_disk_store s(dir, 1ull << 30);   // a restart: the entries are still there
        KR_CHECK(s.entries() == 3);                   // chat A's turns 2 and 3 (re-saved above), chat B
        kera::saved_prompt_blobs out;
        KR_CHECK(s.load_best(seq(60, 1000), 0, 0.25, out) && out.blobs[0][0] == 3);
    }
    fs::remove_all(dir);
}

KR_TEST(eviction_keeps_the_budget) {
    const std::string dir = temp_dir("kera_test_store_budget");
    kera::prompt_disk_store s(dir, 25000);            // room for two 10 KB entries
    s.save_async(make(seq(10, 0), 10000, 1));
    s.flush();
    s.save_async(make(seq(10, 100), 10000, 2));
    s.flush();
    kera::saved_prompt_blobs out;
    KR_CHECK(s.load_best(seq(10, 0), 0, 0.25, out));  // touch the first, so the second is the oldest
    s.save_async(make(seq(10, 200), 10000, 3));
    s.flush();
    KR_CHECK(s.entries() == 2);
    KR_CHECK(s.bytes_on_disk() <= 25000);
    KR_CHECK(s.load_best(seq(10, 0), 0, 0.25, out));
    KR_CHECK(!s.load_best(seq(10, 100), 0, 0.25, out));
    KR_CHECK(s.load_best(seq(10, 200), 0, 0.25, out));
    fs::remove_all(dir);
}

KR_TEST(a_damaged_size_is_rejected_not_allocated) {
    const std::string dir = temp_dir("kera_test_kvp_size");
    fs::create_directories(dir);
    const std::string path = dir + "/a.kvp";
    KR_CHECK(kera::write_saved_prompt(path, make(seq(10), 100, 3)));
    {
        // the first blob's size, after the magic, version, tokens and meta: made enormous
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        f.seekp(8 + 4 + 4 + 10 * 4 + 4 + 3 * 8 + 4);
        const uint64_t huge = 1ull << 39;
        f.write((const char *) &huge, sizeof(huge));
    }
    kera::saved_prompt_blobs q;
    KR_CHECK(!kera::read_saved_prompt(path, q)); // rejected without trying to allocate 512 GB
    fs::remove_all(dir);
}

KR_TEST(the_budget_covers_every_models_folder) {
    const std::string root = temp_dir("kera_test_kv_root");
    for (const char * d : { "a", "b", "mine" }) {
        fs::create_directories(fs::u8path(root) / d);
    }
    // two other models' chats, 3 x 1000 bytes each (blobs of about 1000), and this model's folder
    int k = 0;
    for (const char * d : { "a", "b" }) {
        for (int i = 0; i < 3; ++i) {
            KR_CHECK(kera::write_saved_prompt(root + "/" + d + "/" + std::to_string(k) + ".kvp", make(seq(10, k * 100), 1000, 1)));
            fs::last_write_time(fs::u8path(root + "/" + d + "/" + std::to_string(k) + ".kvp"),
                                fs::file_time_type::clock::now() - std::chrono::hours(100 - k));
            ++k;
        }
    }
    KR_CHECK(kera::write_saved_prompt(root + "/mine/x.kvp", make(seq(10), 1000, 1)));
    // a budget of about 4 files: the others keep at most 2, the oldest go first, and this model's are untouched
    const uint64_t one = (uint64_t) fs::file_size(fs::u8path(root + "/a/0.kvp"));
    const uint64_t others = kera::trim_saved_prompts(root, root + "/mine", 4 * one + one / 2);
    KR_CHECK(others <= 2 * one + one / 4);
    KR_CHECK(!fs::exists(fs::u8path(root + "/a/0.kvp")) && !fs::exists(fs::u8path(root + "/a/2.kvp")));
    KR_CHECK(!fs::exists(fs::u8path(root + "/a"))); // left empty, so removed
    KR_CHECK(fs::exists(fs::u8path(root + "/b/5.kvp")) && fs::exists(fs::u8path(root + "/mine/x.kvp")));
    fs::remove_all(root);
}

KR_TEST_MAIN()
