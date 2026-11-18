// Saved chat states on the SSD (kr_prompt_disk.h).
//
// File format (little-endian), version 1:
//   "KERAKVP1"  u32 version
//   u32 n_tokens  i32 tokens[n_tokens]
//   u32 n_meta    i64 meta[n_meta]
//   u32 n_blobs   { u64 size, bytes[size] } x n_blobs
//   u64 XXH3-64 of everything from the version on

#include "kvstore/kr_prompt_disk.h"

#include "common/kr_hash.h"

#include "hash/xxhash/xxhash.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>

namespace fs = std::filesystem;

namespace kera {

namespace {

// A folder's entries, stopping at the first error: a range-for over directory_iterator throws when reading the next
// entry fails (a folder removed or its permissions changed meanwhile), which would stop kera serve
template <class F> void each_entry(const fs::path & dir, F && f) {
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        f(*it);
    }
}

} // namespace

namespace {

const char MAGIC[8] = { 'K', 'E', 'R', 'A', 'K', 'V', 'P', '1' };
constexpr uint32_t VERSION = 1;

struct hashing_writer {
    std::ofstream & f;
    XXH3_state_t * st;
    void put(const void * p, size_t n) {
        f.write((const char *) p, (std::streamsize) n);
        XXH3_64bits_update(st, p, n);
    }
    template <typename T> void put(T v) { put(&v, sizeof(v)); }
};

struct hashing_reader {
    std::ifstream & f;
    XXH3_state_t * st;
    bool get(void * p, size_t n) {
        if (!f.read((char *) p, (std::streamsize) n)) {
            return false;
        }
        XXH3_64bits_update(st, p, n);
        return true;
    }
    template <typename T> bool get(T & v) { return get(&v, sizeof(v)); }
};

size_t common_prefix(const std::vector<int32_t> & a, const std::vector<int32_t> & b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) {
        ++i;
    }
    return i;
}

} // namespace

uint64_t saved_prompt_blobs::bytes() const {
    uint64_t n = tokens.size() * 4 + meta.size() * 8;
    for (const auto & b : blobs) {
        n += b.size();
    }
    return n;
}

bool write_saved_prompt(const std::string & path, const saved_prompt_blobs & p) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(fs::u8path(tmp), std::ios::binary | std::ios::trunc);
        if (!f) {
            return false;
        }
        XXH3_state_t * st = XXH3_createState();
        XXH3_64bits_reset(st);
        f.write(MAGIC, sizeof(MAGIC));
        hashing_writer w { f, st };
        w.put(VERSION);
        w.put((uint32_t) p.tokens.size());
        w.put(p.tokens.data(), p.tokens.size() * sizeof(int32_t));
        w.put((uint32_t) p.meta.size());
        w.put(p.meta.data(), p.meta.size() * sizeof(int64_t));
        w.put((uint32_t) p.blobs.size());
        for (const auto & b : p.blobs) {
            w.put((uint64_t) b.size());
            w.put(b.data(), b.size());
        }
        const uint64_t h = XXH3_64bits_digest(st);
        XXH3_freeState(st);
        f.write((const char *) &h, sizeof(h));
        f.close(); // the last buffer is written here: its error counts too
        if (!f) {
            std::error_code ec;
            fs::remove(fs::u8path(tmp), ec); // a disk that is full stays no fuller for it
            return false;
        }
    }
    std::error_code ec;
    fs::rename(fs::u8path(tmp), fs::u8path(path), ec);
    if (ec) {
        fs::remove(fs::u8path(tmp), ec);
        return false;
    }
    return true;
}

bool read_saved_prompt(const std::string & path, saved_prompt_blobs & p, bool tokens_only) {
    std::error_code ec;
    // no size in the file is believed beyond the file's own: a damaged one is rejected, not allocated
    const uint64_t file_bytes = (uint64_t) fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return false;
    }
    std::ifstream f(fs::u8path(path), std::ios::binary);
    char magic[8];
    if (!f.read(magic, sizeof(magic)) || std::memcmp(magic, MAGIC, sizeof(MAGIC)) != 0) {
        return false;
    }
    XXH3_state_t * st = XXH3_createState();
    XXH3_64bits_reset(st);
    hashing_reader r { f, st };
    bool ok = true;
    uint32_t version = 0, n = 0;
    ok = ok && r.get(version) && version == VERSION;
    ok = ok && r.get(n) && n < (1u << 28) && (uint64_t) n * 4 <= file_bytes;
    if (ok) {
        p.tokens.resize(n);
        ok = r.get(p.tokens.data(), (size_t) n * sizeof(int32_t));
    }
    if (ok && !tokens_only) {
        ok = r.get(n) && n < (1u << 20);
        ok = ok && (uint64_t) n * 8 <= file_bytes;
        if (ok) {
            p.meta.resize(n);
            ok = r.get(p.meta.data(), (size_t) n * sizeof(int64_t));
        }
        ok = ok && r.get(n) && n < (1u << 20);
        if (ok) {
            p.blobs.assign(n, {});
            for (auto & b : p.blobs) {
                uint64_t size = 0;
                ok = ok && r.get(size) && size <= file_bytes;
                if (!ok) {
                    break;
                }
                b.resize((size_t) size);
                ok = r.get(b.data(), b.size());
            }
        }
        uint64_t want = 0;
        ok = ok && f.read((char *) &want, sizeof(want)) && want == XXH3_64bits_digest(st);
    }
    XXH3_freeState(st);
    return ok;
}

prompt_disk_store::prompt_disk_store(const std::string & dir, uint64_t budget_bytes) : dir_(dir), budget_(budget_bytes) {
    std::error_code ec;
    fs::create_directories(fs::u8path(dir_), ec);
    index_dir();
    thread_ = std::thread([this] { writer(); });
}

prompt_disk_store::~prompt_disk_store() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_.notify_all();
    thread_.join();
}

void prompt_disk_store::index_dir() {
    // a write still in progress in another kera serve with the same options (the app's beside a terminal's) is
    // left alone: only one older than ten minutes never finished
    const auto stale = fs::file_time_type::clock::now() - std::chrono::minutes(10);
    each_entry(fs::u8path(dir_), [&](const fs::directory_entry & de) {
        std::error_code ec;
        const std::string path = de.path().u8string();
        if (de.path().extension() == ".tmp") {
            const auto t = fs::last_write_time(de.path(), ec);
            if (!ec && t < stale) {
                fs::remove(de.path(), ec);
            }
            return;
        }
        if (de.path().extension() != ".kvp") {
            return;
        }
        entry e;
        saved_prompt_blobs p;
        if (!read_saved_prompt(path, p, true)) {
            fs::remove(de.path(), ec);
            return;
        }
        const uintmax_t size = fs::file_size(de.path(), ec);
        if (ec) {
            return; // gone meanwhile (another store trimmed it)
        }
        e.file = path;
        e.tokens = std::move(p.tokens);
        e.size = (uint64_t) size;
        e.last_used = (uint64_t) fs::last_write_time(de.path(), ec).time_since_epoch().count();
        index_.push_back(std::move(e));
    });
    // keep last_used as an ordering clock that continues after the newest file
    for (const auto & e : index_) {
        clock_ = std::max(clock_, e.last_used);
    }
    evict_locked(); // a smaller budget than last time applies at once (no other thread runs yet)
}

void prompt_disk_store::save_async(saved_prompt_blobs && p) {
    if (p.tokens.empty()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        // a newer state for the same chat supersedes one still waiting to be written
        queue_.erase(std::remove_if(queue_.begin(), queue_.end(), [&](const saved_prompt_blobs & q) {
            return q.tokens.size() <= p.tokens.size() && common_prefix(q.tokens, p.tokens) == q.tokens.size();
        }), queue_.end());
        queue_.push_back(std::move(p));
    }
    cv_.notify_all();
}

void prompt_disk_store::writer() {
    std::unique_lock<std::mutex> lock(mu_);
    while (true) {
        cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
        if (queue_.empty()) {
            if (stop_) {
                break;
            }
            continue;
        }
        saved_prompt_blobs p = std::move(queue_.front());
        queue_.pop_front();
        busy_ = true;
        lock.unlock();
        write_now(p);
        lock.lock();
        busy_ = false;
        idle_cv_.notify_all();
    }
}

void prompt_disk_store::write_now(saved_prompt_blobs & p) {
    std::error_code dir_ec;
    fs::create_directories(fs::u8path(dir_), dir_ec); // in case another store's clean-up took it
    const std::string path = (fs::u8path(dir_) / (hash_bytes(p.tokens.data(), p.tokens.size() * 4) + ".kvp")).u8string();
    if (!write_saved_prompt(path, p)) {
        return;
    }
    std::error_code ec;
    const uintmax_t size_or_error = fs::file_size(fs::u8path(path), ec);
    if (ec) {
        return; // removed just now by another store's clean-up: nothing to index
    }
    const uint64_t size = (uint64_t) size_or_error;
    std::lock_guard<std::mutex> lock(mu_);
    // drop entries this one covers: the same chat's earlier turns share almost all their tokens with it
    // (the last answer is often written back slightly differently, so they aren't an exact start of it)
    for (auto it = index_.begin(); it != index_.end();) {
        if (it->tokens.size() <= p.tokens.size() && common_prefix(it->tokens, p.tokens) >= it->tokens.size() * 9 / 10) {
            if (it->file != path) {
                fs::remove(fs::u8path(it->file), ec);
            }
            it = index_.erase(it);
        } else {
            ++it;
        }
    }
    index_.push_back({ path, p.tokens, size, ++clock_ });
    evict_locked();
}

void prompt_disk_store::evict_locked() {
    uint64_t total = 0;
    for (const auto & e : index_) {
        total += e.size;
    }
    std::error_code ec;
    while (total > budget_ && !index_.empty()) {
        auto oldest = std::min_element(index_.begin(), index_.end(),
                                       [](const entry & a, const entry & b) { return a.last_used < b.last_used; });
        total -= oldest->size;
        fs::remove(fs::u8path(oldest->file), ec);
        index_.erase(oldest);
    }
}

bool prompt_disk_store::load_best(const std::vector<int32_t> & tokens, size_t min_lcp, double min_keep,
                                  saved_prompt_blobs & out, size_t * lcp_out) {
    std::string file;
    size_t best = 0;
    {
        std::lock_guard<std::mutex> lock(mu_);
        entry * pick = nullptr;
        for (auto & e : index_) {
            const size_t lcp = common_prefix(e.tokens, tokens);
            if (lcp > min_lcp && lcp > best && (double) lcp >= min_keep * (double) e.tokens.size()) {
                best = lcp;
                pick = &e;
            }
        }
        if (pick == nullptr) {
            return false;
        }
        pick->last_used = ++clock_;
        file = pick->file;
    }
    if (!read_saved_prompt(file, out)) {
        std::lock_guard<std::mutex> lock(mu_);
        std::error_code ec;
        fs::remove(fs::u8path(file), ec);
        index_.erase(std::remove_if(index_.begin(), index_.end(), [&](const entry & e) { return e.file == file; }), index_.end());
        return false;
    }
    if (lcp_out) {
        *lcp_out = best;
    }
    return true;
}

void prompt_disk_store::flush() {
    std::unique_lock<std::mutex> lock(mu_);
    idle_cv_.wait(lock, [&] { return queue_.empty() && !busy_; });
}

size_t prompt_disk_store::entries() const {
    std::lock_guard<std::mutex> lock(mu_);
    return index_.size();
}

uint64_t prompt_disk_store::bytes_on_disk() const {
    std::lock_guard<std::mutex> lock(mu_);
    uint64_t n = 0;
    for (const auto & e : index_) {
        n += e.size;
    }
    return n;
}

uint64_t trim_saved_prompts(const std::string & root, const std::string & keep, uint64_t budget) {
    struct file {
        fs::path path;
        uint64_t size;
        fs::file_time_type time;
    };
    std::vector<file> files;
    uint64_t total = 0;
    std::error_code ec;
    const fs::path keep_path = fs::u8path(keep);
    each_entry(fs::u8path(root), [&](const fs::directory_entry & d) {
        if (!d.is_directory(ec) || fs::equivalent(d.path(), keep_path, ec)) {
            return;
        }
        each_entry(d.path(), [&](const fs::directory_entry & de) {
            if (de.path().extension() == ".kvp") {
                std::error_code size_ec;
                const uintmax_t size = fs::file_size(de.path(), size_ec);
                if (!size_ec) {
                    files.push_back({ de.path(), (uint64_t) size, fs::last_write_time(de.path(), size_ec) });
                    total += size;
                }
            }
        });
    });
    // the oldest first, until the others take at most half the budget
    std::sort(files.begin(), files.end(), [](const file & a, const file & b) { return a.time < b.time; });
    std::set<fs::path> emptied;
    for (const auto & f : files) {
        if (total <= budget / 2) {
            break;
        }
        if (fs::remove(f.path, ec)) {
            total -= f.size;
            emptied.insert(f.path.parent_path());
        }
    }
    // folders this left empty go too (a save makes its folder again). One that was empty already stays a day:
    // another kera serve running now (the other engine of --backend auto, or the app's beside a terminal's) may
    // have just opened it
    const auto day_ago = fs::file_time_type::clock::now() - std::chrono::hours(24);
    std::vector<fs::path> empty_dirs;
    each_entry(fs::u8path(root), [&](const fs::directory_entry & d) {
        if (d.is_directory(ec) && !fs::equivalent(d.path(), keep_path, ec) && fs::is_empty(d.path(), ec) &&
            (emptied.count(d.path()) > 0 || fs::last_write_time(d.path(), ec) < day_ago)) {
            empty_dirs.push_back(d.path());
        }
    });
    for (const auto & d : empty_dirs) {
        fs::remove(d, ec); // after the scan: removing entries while iterating is unspecified
    }
    return total;
}

} // namespace kera
