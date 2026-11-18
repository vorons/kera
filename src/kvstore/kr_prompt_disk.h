// Saved chat states on the SSD (spec 4b item 5, first version): whole prompt states keyed by their
// tokens, found again by the longest shared start. llama.cpp's server keeps recent chats in RAM; this
// is the tier behind it, so old chats and restarts don't mean reading the whole chat again.
//
// What a saved prompt holds is opaque here (the server adapter decides): its tokens, a few numbers
// and some byte blobs. Files carry a checksum, are written in the background, and are evicted
// least-recently-used first when the store passes its disk budget.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace kera {

struct saved_prompt_blobs {
    std::vector<int32_t> tokens;
    std::vector<int64_t> meta;
    std::vector<std::vector<uint8_t>> blobs;

    uint64_t bytes() const;
};

class prompt_disk_store {
public:
    // dir is created if missing; entries already there are indexed (their tokens are read).
    prompt_disk_store(const std::string & dir, uint64_t budget_bytes);
    ~prompt_disk_store(); // finishes the writes still queued

    // Queue a save. Entries this one covers are removed: the same tokens, or at least 90% of an
    // entry's tokens shared as a start (the same chat's earlier turns).
    void save_async(saved_prompt_blobs && p);

    // The saved prompt sharing the longest start with `tokens`, if that start is longer than min_lcp
    // and covers at least min_keep of the saved prompt (loading a huge state for a small start
    // isn't worth it). A file that fails its checksum is deleted.
    bool load_best(const std::vector<int32_t> & tokens, size_t min_lcp, double min_keep,
                   saved_prompt_blobs & out, size_t * lcp = nullptr);

    void flush();               // wait until the queue is written
    size_t entries() const;
    uint64_t bytes_on_disk() const;

private:
    struct entry {
        std::string file;
        std::vector<int32_t> tokens;
        uint64_t size = 0;
        uint64_t last_used = 0;
    };

    void writer();
    void write_now(saved_prompt_blobs & p);
    void evict_locked();
    void index_dir();

    std::string dir_;
    uint64_t budget_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable idle_cv_;
    std::deque<saved_prompt_blobs> queue_;
    std::vector<entry> index_;
    uint64_t clock_ = 0;
    bool busy_ = false;
    bool stop_ = false;
    std::thread thread_;
};

// The disk budget covers every model's saved chats together: each model's options get a folder of their own under
// root, so before a store opens its folder (keep), the other folders' oldest files are removed until they hold at
// most half the budget. Returns what they still hold; the store gets the rest.
uint64_t trim_saved_prompts(const std::string & root, const std::string & keep, uint64_t budget);

// The file format, exposed for tests: write/read one saved prompt.
bool write_saved_prompt(const std::string & path, const saved_prompt_blobs & p);
bool read_saved_prompt(const std::string & path, saved_prompt_blobs & p, bool tokens_only = false);

} // namespace kera
