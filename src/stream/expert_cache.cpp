// Kera's expert cache (expert_cache.h).

#include "stream/expert_cache.h"

#include <cstdio>
#include <cstring>

#include <algorithm>
#include <chrono>

namespace kera::stream {

namespace {
using clock_type = std::chrono::steady_clock;
}

expert_cache::~expert_cache() {
    close();
}

bool expert_cache::open(const std::vector<std::string> & files, const std::vector<uint64_t> & file_base,
                        const std::vector<std::vector<std::string>> & mirrors, region * mirror,
                        const std::vector<layer_layout> & layers, const cache_options & opt, std::string * error) {
    close();
    auto fail = [&](const std::string & msg) {
        if (error) {
            *error = msg;
        }
        close();
        return false;
    };
    if (mirror == nullptr || !mirror->reserved() || files.size() != file_base.size()) {
        return fail("the mirror of the model's files isn't set up");
    }
    page_ = region::page_size();
    opt_ = opt;
    opt_.chunk_bytes = (uint32_t) std::max<uint64_t>(page_, opt_.chunk_bytes / page_ * page_);
    opt_.io_threads = std::max(1, opt_.io_threads);
    files_ = files;
    file_base_ = file_base;
    mirror_ = mirror;
    layers_ = layers;

    copies_.assign(files.size(), {});
    inflight_.assign(files.size(), {});
    for (size_t i = 0; i < files.size(); ++i) {
        os::file f;
        const int err = os::open_read(files[i], opt_.direct, f);
        if (err != 0) {
            return fail("can't open " + files[i] + " (error " + std::to_string(err) + ")");
        }
        copies_[i].push_back(f);
        inflight_[i].push_back(0);
        if (i < mirrors.size()) {
            for (const auto & m : mirrors[i]) {
                os::file g;
                if (os::open_read(m, opt_.direct, g) == 0) {
                    copies_[i].push_back(g);
                    inflight_[i].push_back(0);
                }
            }
        }
    }

    int max_layer = 0;
    for (const auto & l : layers_) {
        max_layer = std::max(max_layer, l.layer);
    }
    layer_index_.assign((size_t) max_layer + 1, -1);
    first_id_.clear();
    expert_bytes_.clear();
    int32_t n = 0;
    for (size_t li = 0; li < layers_.size(); ++li) {
        layer_index_[(size_t) layers_[li].layer] = (int) li;
        first_id_.push_back(n);
        n += layers_[li].n_expert;
    }
    experts_.assign((size_t) n, expert{});
    std::vector<span> sp;
    for (size_t li = 0; li < layers_.size(); ++li) {
        uint64_t biggest = 0;
        for (int e = 0; e < layers_[li].n_expert; ++e) {
            const int32_t id = first_id_[li] + e;
            spans_of(id, sp);
            expert & ex = experts_[(size_t) id];
            for (const auto & s : sp) {
                if (s.hi > mirror_->size()) {
                    return fail("an expert lies outside the model's files");
                }
                ex.bytes += s.hi - s.lo;
                ex.chunks_total += (uint32_t) ((s.hi - s.lo + opt_.chunk_bytes - 1) / opt_.chunk_bytes);
            }
            biggest = std::max(biggest, ex.bytes);
        }
        expert_bytes_.push_back(biggest);
    }

    st_ = cache_stats{};
    st_.budget_bytes = opt_.budget_bytes;
    stop_ = false;
    for (int i = 0; i < opt_.io_threads; ++i) {
        threads_.emplace_back([this] { worker(); });
    }
    return true;
}

void expert_cache::close() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto & t : threads_) {
        t.join();
    }
    threads_.clear();
    for (auto & c : copies_) {
        for (auto & f : c) {
            os::close(f);
        }
    }
    copies_.clear();
    inflight_.clear();
    for (auto & q : queues_) {
        q.clear();
    }
    experts_.clear();
    edge_refs_.clear();
    lru_head_ = lru_tail_ = -1;
    used_ = pinned_ = 0;
    mirror_ = nullptr;
}

int expert_cache::id_of(int layer, int e) const {
    if (layer < 0 || (size_t) layer >= layer_index_.size()) {
        return -1;
    }
    const int li = layer_index_[(size_t) layer];
    if (li < 0 || e < 0 || e >= layers_[(size_t) li].n_expert) {
        return -1;
    }
    return first_id_[(size_t) li] + e;
}

void expert_cache::spans_of(int32_t id, std::vector<span> & out) const {
    out.clear();
    const size_t li = (size_t) (std::upper_bound(first_id_.begin(), first_id_.end(), id) - first_id_.begin() - 1);
    const uint64_t e = (uint64_t) (id - first_id_[li]);
    for (const auto & p : layers_[li].parts) {
        const uint64_t start = file_base_[(size_t) p.file] + p.file_offset + e * p.stride;
        const uint64_t end = start + p.stride;
        out.push_back({ start / page_ * page_, (end + page_ - 1) / page_ * page_, start, end });
    }
}

void expert_cache::lru_remove(int32_t id) {
    expert & ex = experts_[(size_t) id];
    if (ex.prev < 0 && lru_head_ != id) {
        return; // not in the list
    }
    if (ex.prev >= 0) {
        experts_[(size_t) ex.prev].next = ex.next;
    } else {
        lru_head_ = ex.next;
    }
    if (ex.next >= 0) {
        experts_[(size_t) ex.next].prev = ex.prev;
    } else {
        lru_tail_ = ex.prev;
    }
    ex.prev = ex.next = -1;
}

void expert_cache::lru_push(int32_t id) {
    expert & ex = experts_[(size_t) id];
    ex.prev = lru_tail_;
    ex.next = -1;
    if (lru_tail_ >= 0) {
        experts_[(size_t) lru_tail_].next = id;
    } else {
        lru_head_ = id;
    }
    lru_tail_ = id;
}

bool expert_cache::make_room(uint64_t bytes, bool force) {
    if (!force && bytes > 0) {
        // a guess that can't fit even with every unlocked expert gone evicts nothing: emptying the cache for a read
        // that then doesn't happen would only make the next layer miss everything (with bytes = 0, the call that
        // brings the cache back within its budget, everything unlocked may go)
        uint64_t evictable = 0;
        for (int32_t i = lru_head_; i >= 0; i = experts_[(size_t) i].next) {
            if (experts_[(size_t) i].lock != epoch_) {
                evictable += experts_[(size_t) i].bytes;
            }
        }
        if (used_ + bytes > opt_.budget_bytes + evictable) {
            return false;
        }
    }
    int32_t id = lru_head_;
    while (used_ + bytes > opt_.budget_bytes && id >= 0) {
        const int32_t next = experts_[(size_t) id].next;
        if (experts_[(size_t) id].lock != epoch_) {
            evict(id);
        }
        id = next;
    }
    return used_ + bytes <= opt_.budget_bytes || force;
}

void expert_cache::evict(int32_t id) {
    expert & ex = experts_[(size_t) id];
    lru_remove(id);
    ex.state = ABSENT;
    ex.early = false;
    used_ -= ex.bytes;
    st_.evictions++;
    release_pages(id);
}

// an expert's pages go back to the OS, and the pages it shares with a neighbour once neither holds them
void expert_cache::release_pages(int32_t id) {
    std::vector<span> sp;
    spans_of(id, sp);
    for (const auto & s : sp) {
        if (s.hi - s.lo > 2 * page_) {
            mirror_->decommit(s.lo + page_, s.hi - s.lo - 2 * page_);
        }
        for (const uint64_t p : { s.lo, s.hi - page_ }) {
            auto it = edge_refs_.find(p);
            if (it != edge_refs_.end() && --it->second == 0) {
                edge_refs_.erase(it);
                mirror_->decommit(p, page_);
            }
            if (s.hi - s.lo == page_) {
                break; // one page: both edges are the same page
            }
        }
    }
}

void expert_cache::start_load(int32_t id, uint8_t q) {
    expert & ex = experts_[(size_t) id];
    ex.state = QUEUED;
    ex.failed = false;
    ex.chunks_left = ex.chunks_total;
    ex.chunks_started = 0;
    used_ += ex.bytes;
    std::vector<span> sp;
    spans_of(id, sp);
    const size_t li = (size_t) (std::upper_bound(first_id_.begin(), first_id_.end(), id) - first_id_.begin() - 1);
    for (size_t k = 0; k < sp.size(); ++k) {
        const span & s = sp[k];
        edge_refs_[s.lo]++;
        if (s.hi - s.lo > page_) {
            edge_refs_[s.hi - page_]++;
        }
        const int file = layers_[li].parts[k].file;
        for (uint64_t off = s.lo; off < s.hi; off += opt_.chunk_bytes) {
            queues_[q].push_back({ id, file, off, (uint32_t) std::min<uint64_t>(opt_.chunk_bytes, s.hi - off), s.own_lo,
                                   s.own_hi });
        }
    }
    cv_work_.notify_all();
}

void expert_cache::promote(int32_t id) {
    for (int q = Q_AHEAD; q <= Q_BACKGROUND; ++q) {
        auto & from = queues_[q];
        for (auto it = from.begin(); it != from.end();) {
            if (it->id == id) {
                queues_[Q_DEMAND].push_back(*it);
                it = from.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void expert_cache::cancel_guesses(int li, const std::vector<uint8_t> & wanted) {
    auto & q = queues_[Q_AHEAD];
    if (q.empty()) {
        return;
    }
    const int32_t lo = first_id_[(size_t) li];
    const int32_t hi = lo + layers_[(size_t) li].n_expert;
    std::vector<int32_t> cancelled;
    std::deque<chunk> keep;
    for (const chunk & c : q) {
        const bool this_layer = c.id >= lo && c.id < hi;
        if (this_layer && !wanted[(size_t) (c.id - lo)] && experts_[(size_t) c.id].chunks_started == 0) {
            if (cancelled.empty() || cancelled.back() != c.id) {
                cancelled.push_back(c.id);
            }
            continue;
        }
        keep.push_back(c);
    }
    q.swap(keep);
    std::sort(cancelled.begin(), cancelled.end());
    cancelled.erase(std::unique(cancelled.begin(), cancelled.end()), cancelled.end());
    std::vector<span> sp;
    for (const int32_t id : cancelled) {
        expert & ex = experts_[(size_t) id];
        ex.state = ABSENT;
        ex.early = false;
        used_ -= ex.bytes;
        st_.prefetch_cancelled++;
        spans_of(id, sp);
        for (const auto & s : sp) {
            for (const uint64_t p : { s.lo, s.hi - page_ }) {
                auto it = edge_refs_.find(p);
                if (it != edge_refs_.end() && --it->second == 0) {
                    edge_refs_.erase(it);
                    mirror_->decommit(p, page_);
                }
                if (s.hi - s.lo == page_) {
                    break;
                }
            }
        }
    }
}

int expert_cache::pick_copy(int file) {
    const auto & v = inflight_[(size_t) file];
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i) {
        if (v[i] < v[(size_t) best]) {
            best = (int) i;
        }
    }
    return best;
}

// Reads one chunk into the mirror. A chunk's first and last page can hold a neighbouring part's bytes too: such a page
// is read into this worker's own page (bounce) and only this part's bytes are copied in, so a read never writes a
// neighbour's bytes. Two parts that share a page are often read at once, and an unbuffered read on an encrypted drive
// (BitLocker, dm-crypt) lands as ciphertext and is decrypted where it landed: on the Minisforum the page two such
// reads shared now and then ended up wrong, and so did the experts in it (docs/NEGATIVE_RESULTS.md N-012).
int expert_cache::read_chunk(const chunk & c, int copy, uint8_t * bounce, size_t * done) {
    const os::file & f = copies_[(size_t) c.file][(size_t) copy];
    const uint64_t base = file_base_[(size_t) c.file];
    uint8_t * m = mirror_->base();
    uint64_t lo = c.mirror_off;
    uint64_t hi = c.mirror_off + c.len;
    *done = 0;
    auto shared_page = [&](uint64_t page) {
        size_t got = 0;
        const int st = os::pread_full(f, page - base, (size_t) page_, bounce, &got);
        if (st != 0) {
            return st;
        }
        if (page + (uint64_t) got < std::min(page + page_, c.own_hi)) {
            // the part's own bytes on this page lie inside the file: fewer of them than asked is a failed read (a file
            // cut short while Kera runs), not the file's end, as for the pages below (N-018)
            return k_short_read;
        }
        const uint64_t a = std::max(page, c.own_lo);
        const uint64_t b = std::min(std::min(page + page_, c.own_hi), page + (uint64_t) got);
        if (b > a) {
            std::memcpy(m + a, bounce + (a - page), (size_t) (b - a));
        }
        *done += got;
        return 0;
    };
    if (lo < c.own_lo) { // the first page starts with a neighbour's bytes
        if (const int st = shared_page(lo)) {
            return st;
        }
        lo += page_;
    }
    if (hi > lo && hi > c.own_hi) { // the last page ends with a neighbour's bytes
        if (const int st = shared_page(hi - page_)) {
            return st;
        }
        hi -= page_;
    }
    if (hi > lo) { // the pages only this part has: straight into the mirror
        size_t got = 0;
        const int st = os::pread_full(f, lo - base, (size_t) (hi - lo), m + lo, &got);
        *done += got;
        // these pages lie inside the file: fewer bytes than asked is a failed read, not the file's end (the expert
        // would otherwise be used with zeros where its last bytes should be)
        return st != 0 ? st : got == (size_t) (hi - lo) ? 0 : k_short_read;
    }
    return 0;
}

void expert_cache::worker() {
    uint8_t * bounce = (uint8_t *) os::aligned_alloc((size_t) page_, (size_t) page_);
    struct freer {
        uint8_t * p;
        ~freer() { os::aligned_free(p); }
    } free_bounce{ bounce };
    for (;;) {
        chunk c{};
        int copy = 0;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_work_.wait(lock, [&] {
                return stop_ || !queues_[0].empty() || !queues_[1].empty() || !queues_[2].empty();
            });
            if (stop_) {
                return;
            }
            for (auto & q : queues_) {
                if (!q.empty()) {
                    c = q.front();
                    q.pop_front();
                    break;
                }
            }
            expert & ex = experts_[(size_t) c.id];
            if (ex.state == QUEUED) {
                ex.state = LOADING;
            }
            ex.chunks_started++;
            copy = pick_copy(c.file);
            inflight_[(size_t) c.file][(size_t) copy] += c.len;
        }

        const uint64_t file_off = c.mirror_off - file_base_[(size_t) c.file];
        size_t done = 0;
        int status = mirror_->commit(c.mirror_off, c.len) ? 0 : -1;
        if (status == 0 && bounce == nullptr) {
            status = -1; // no memory for the shared pages
        }
        if (status == 0) {
            status = read_chunk(c, copy, bounce, &done);
            if (status != 0 && copy != 0) {
                // a mirror failed: read the original instead (Phase 5 falls back to the main drive)
                status = read_chunk(c, 0, bounce, &done);
            }
            // a read can fail for a moment (on Windows, when the pages it locks are short): say so and try again
            for (int attempt = 1; status != 0 && attempt <= 3; ++attempt) {
                std::fprintf(stderr, "Kera: reading %zu bytes of %s at %llu failed (error %d); trying again (%d of 3)\n",
                             (size_t) c.len, files_[(size_t) c.file].c_str(), (unsigned long long) file_off, status, attempt);
                std::this_thread::sleep_for(std::chrono::milliseconds(20 * attempt));
                status = read_chunk(c, 0, bounce, &done);
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            inflight_[(size_t) c.file][(size_t) copy] -= c.len;
            st_.bytes_read += done;
            expert & ex = experts_[(size_t) c.id];
            if (status != 0) {
                // this expert only: a guess that fails stops nothing, and the next request for it reads it again
                ex.failed = true;
                error_msg_ = status < 0 ? "out of memory for an expert"
                           : status == k_short_read ? "a read of " + files_[(size_t) c.file] + " ended early"
                           : "a read of " + files_[(size_t) c.file] + " failed (error " + std::to_string(status) + ")";
            }
            if (--ex.chunks_left == 0) {
                if (ex.failed) {
                    ex.state = ABSENT;
                    used_ -= ex.bytes;
                    release_pages(c.id);
                } else {
                    ex.state = RESIDENT;
                    if (ex.pin == 0) {
                        lru_push(c.id);
                    }
                }
                cv_done_.notify_all();
            }
        }
    }
}

bool expert_cache::require(int layer, const int32_t * ids, size_t n) {
    std::unique_lock<std::mutex> lock(mutex_);
    epoch_++;
    st_.steps++;
    if (layer < 0 || (size_t) layer >= layer_index_.size() || layer_index_[(size_t) layer] < 0) {
        return true;
    }
    const int li = layer_index_[(size_t) layer];
    const int n_exp = layers_[(size_t) li].n_expert;
    std::vector<uint8_t> wanted((size_t) n_exp, 0);
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] >= 0 && ids[i] < n_exp) {
            wanted[(size_t) ids[i]] = 1;
        }
    }
    cancel_guesses(li, wanted);

    std::vector<int32_t> waiting;
    std::vector<int32_t> to_load;
    uint64_t need = 0;
    for (int e = 0; e < n_exp; ++e) {
        if (!wanted[(size_t) e]) {
            continue;
        }
        const int32_t id = first_id_[(size_t) li] + e;
        expert & ex = experts_[(size_t) id];
        ex.lock = epoch_;
        ex.uses++;
        st_.requests++;
        need += ex.bytes;
        if (ex.early && ex.state != ABSENT) {
            st_.prefetch_used++;
            ex.early = false;
        }
        switch (ex.state) {
            case RESIDENT:
                st_.hits++;
                if (ex.pin == 0) {
                    lru_remove(id);
                    lru_push(id);
                }
                break;
            case QUEUED:
            case LOADING:
                st_.late++;
                promote(id);
                waiting.push_back(id);
                break;
            default:
                st_.misses++;
                to_load.push_back(id);
                break;
        }
    }
    if (need > opt_.budget_bytes) {
        st_.overflow_steps++;
    }
    for (const int32_t id : to_load) {
        make_room(experts_[(size_t) id].bytes, /*force =*/ true);
        start_load(id, Q_DEMAND);
        waiting.push_back(id);
    }
    make_room(0, false); // back within the budget once the last layer's experts are unlocked

    bool ok = true;
    if (!waiting.empty()) {
        const auto t0 = clock_type::now();
        // each one in memory, or failed (its reads tried again and still failing)
        cv_done_.wait(lock, [&] {
            for (const int32_t id : waiting) {
                const expert & ex = experts_[(size_t) id];
                if (ex.state != RESIDENT && !(ex.state == ABSENT && ex.failed)) {
                    return false;
                }
            }
            return true;
        });
        st_.io_wait_s += std::chrono::duration<double>(clock_type::now() - t0).count();
        for (const int32_t id : waiting) {
            ok = ok && experts_[(size_t) id].state == RESIDENT;
        }
    }
    return ok;
}

void expert_cache::prefetch(int layer, const int32_t * ids, size_t n, reason why) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
        const int id = id_of(layer, ids[i]);
        if (id < 0) {
            continue;
        }
        expert & ex = experts_[(size_t) id];
        if (ex.state != ABSENT) {
            continue;
        }
        if (!make_room(ex.bytes, false)) {
            break; // no room for guesses
        }
        ex.early = true;
        st_.prefetched++;
        start_load(id, why == reason::lookahead ? Q_AHEAD : Q_BACKGROUND);
    }
}

uint64_t expert_cache::pin(int layer, const int32_t * ids, size_t n, bool prompt) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t i = 0; i < n; ++i) {
        const int id = id_of(layer, ids[i]);
        if (id < 0) {
            continue;
        }
        expert & ex = experts_[(size_t) id];
        if (ex.pin != 0) {
            continue;
        }
        ex.pin = prompt ? 2 : 1;
        pinned_ += ex.bytes;
        if (ex.state == RESIDENT) {
            lru_remove(id);
        } else if (ex.state == ABSENT && make_room(ex.bytes, false)) {
            ex.early = true;
            st_.prefetched++;
            start_load(id, Q_BACKGROUND);
        }
    }
    return pinned_;
}

void expert_cache::unpin_prompt() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (size_t id = 0; id < experts_.size(); ++id) {
        expert & ex = experts_[id];
        if (ex.pin == 2) {
            ex.pin = 0;
            pinned_ -= ex.bytes;
            if (ex.state == RESIDENT) {
                lru_push((int32_t) id);
            }
        }
    }
}

std::string expert_cache::error_message() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return error_msg_;
}

bool expert_cache::is_resident(int layer, int e) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const int id = id_of(layer, e);
    return id >= 0 && experts_[(size_t) id].state == RESIDENT;
}

bool expert_cache::streams(int layer) const {
    return layer >= 0 && (size_t) layer < layer_index_.size() && layer_index_[(size_t) layer] >= 0;
}

uint64_t expert_cache::expert_bytes(int layer) const {
    return streams(layer) ? expert_bytes_[(size_t) layer_index_[(size_t) layer]] : 0;
}

int expert_cache::n_expert(int layer) const {
    return streams(layer) ? layers_[(size_t) layer_index_[(size_t) layer]].n_expert : 0;
}

std::vector<uint32_t> expert_cache::usage(int layer) const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<uint32_t> out;
    if (!streams(layer)) {
        return out;
    }
    const int li = layer_index_[(size_t) layer];
    for (int e = 0; e < layers_[(size_t) li].n_expert; ++e) {
        out.push_back(experts_[(size_t) (first_id_[(size_t) li] + e)].uses);
    }
    return out;
}

cache_stats expert_cache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_stats s = st_;
    s.used_bytes = used_;
    s.pinned_bytes = pinned_;
    return s;
}

const uint8_t * expert_cache::part_data(int layer, int e, int part) const {
    const int li = streams(layer) ? layer_index_[(size_t) layer] : -1;
    if (li < 0 || part < 0 || (size_t) part >= layers_[(size_t) li].parts.size()) {
        return nullptr;
    }
    const part_layout & p = layers_[(size_t) li].parts[(size_t) part];
    return mirror_->base() + file_base_[(size_t) p.file] + p.file_offset + (uint64_t) e * p.stride;
}

} // namespace kera::stream
