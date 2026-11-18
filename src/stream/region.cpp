// The mirror's address space (region.h), one implementation per OS family.

#include "stream/region.h"

#include <cstdlib>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace kera::stream {

size_t region::page_size() {
    static const size_t ps = [] {
#if defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
#else
        const long p = sysconf(_SC_PAGESIZE);
        return p > 0 ? (size_t) p : (size_t) 4096;
#endif
    }();
    return ps;
}

region::~region() {
    release();
}

bool region::reserve(uint64_t bytes) {
    release();
    const uint64_t ps = page_size();
    bytes = (bytes + ps - 1) / ps * ps;
    if (bytes == 0) {
        return false;
    }
#if defined(_WIN32)
    void * p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE, PAGE_NOACCESS);
    if (p == nullptr) {
        return false;
    }
#else
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_NORESERVE)
    flags |= MAP_NORESERVE; // address space only: RAM is used page by page as reads land
#endif
    void * p = mmap(nullptr, (size_t) bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
    if (p == MAP_FAILED) {
        return false;
    }
#endif
    base_ = (uint8_t *) p;
    size_ = bytes;
    return true;
}

void region::release() {
    if (base_ == nullptr) {
        return;
    }
#if defined(_WIN32)
    VirtualFree(base_, 0, MEM_RELEASE);
#else
    munmap(base_, (size_t) size_);
#endif
    base_ = nullptr;
    size_ = 0;
}

bool region::commit(uint64_t off, uint64_t len) {
    if (len == 0) {
        return true;
    }
#if defined(_WIN32)
    return VirtualAlloc(base_ + off, (SIZE_T) len, MEM_COMMIT, PAGE_READWRITE) != nullptr;
#else
    (void) off;
    return true;
#endif
}

void region::decommit(uint64_t off, uint64_t len) {
    // KERA_STREAM_KEEP=1, for tests only: pages stay as they are once used (docs/NEGATIVE_RESULTS.md N-012)
    static const bool keep = std::getenv("KERA_STREAM_KEEP") != nullptr;
    if (len == 0 || keep) {
        return;
    }
#if defined(_WIN32)
    VirtualFree(base_ + off, (SIZE_T) len, MEM_DECOMMIT);
#elif defined(__linux__)
    madvise(base_ + off, (size_t) len, MADV_DONTNEED); // anonymous pages: freed now, zero when touched again
#else
    // macOS keeps MADV_DONTNEED pages around; mapping fresh zero pages over them frees them at once
    mmap(base_ + off, (size_t) len, PROT_READ | PROT_WRITE, MAP_FIXED | MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
}

} // namespace kera::stream
