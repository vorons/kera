// Internal: the per-OS file primitives behind kr_io and the SSD speed test.
// One implementation per OS: io_linux.cpp, io_macos.cpp, io_windows.cpp.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace kera::os {

#if defined(_WIN32)
using file_handle = void *; // HANDLE
#else
using file_handle = int;
#endif

struct file {
    file_handle h;
    bool direct = false; // true if the OS file cache is bypassed
    bool valid = false;
};

// Open for reading. direct = bypass the OS file cache if the file system allows it; if it doesn't,
// the file is opened normally and f.direct is false. Returns 0 or an OS error code.
int open_read(const std::string & path, bool direct, file & f);

// Create (or truncate) for writing, bypassing the cache when direct is set.
int create_write(const std::string & path, bool direct, file & f);

// Positional read/write of exactly len bytes (less only at end of file). Safe to call from several
// threads on one file. Returns 0 or an OS error code; *done gets the bytes moved.
int pread_full(const file & f, uint64_t offset, size_t len, void * dst, size_t * done);
int pwrite_full(const file & f, uint64_t offset, size_t len, const void * src, size_t * done);

int  sync(const file & f);   // flush to the drive
void close(file & f);
int  remove_file(const std::string & path);
int  make_dir(const std::string & path); // 0 if it exists or was created

void * aligned_alloc(size_t bytes, size_t alignment);
void   aligned_free(void * p);

const char * io_method_name();

} // namespace kera::os
