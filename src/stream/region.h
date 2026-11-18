// A reserved range of address space that mirrors a model's files byte for byte: the data at file offset
// x of file i sits at base + file_base(i) + x. Nothing is in RAM until the store reads it in, so the
// mirror of a 120 GB model costs only the experts Kera keeps. Pages are committed on Windows before a
// read lands in them and given back to the OS when the store evicts an expert (D-012).
#pragma once

#include <cstddef>
#include <cstdint>

namespace kera::stream {

class region {
public:
    region() = default;
    ~region();
    region(const region &) = delete;
    region & operator=(const region &) = delete;

    // Reserve bytes of address space (rounded up to whole pages). Returns false if the OS refuses.
    bool reserve(uint64_t bytes);
    void release();

    uint8_t * base() const { return base_; }
    uint64_t size() const { return size_; }
    bool reserved() const { return base_ != nullptr; }

    // Make [off, off + len) writable before a read lands in it (Windows commits the pages; elsewhere the
    // OS does it on first touch). off and len must be whole pages. Returns false if the OS refuses.
    bool commit(uint64_t off, uint64_t len);

    // Give [off, off + len) back to the OS. The next read of it sees zeros until it is filled again.
    void decommit(uint64_t off, uint64_t len);

    // The OS page size: the unit of commit/decommit and of the reads into the mirror.
    static size_t page_size();

private:
    uint8_t * base_ = nullptr;
    uint64_t size_ = 0;
};

} // namespace kera::stream
