/* I/O back-end (spec P3, section 4): one per OS, big aligned direct reads with completion callbacks.
 *
 * Reads bypass the OS file cache where the OS allows it (O_DIRECT on Linux, F_NOCACHE on macOS,
 * FILE_FLAG_NO_BUFFERING on Windows), so speeds are the drive's own and the RAM budget stays
 * Kera's to manage. Direct reads need aligned offsets, lengths and buffers: use KR_IO_ALIGN.
 *
 * Frozen in M0. Each OS has its own source file: io_linux.cpp, io_macos.cpp, io_windows.cpp.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Offsets, lengths and buffers of direct reads must be multiples of this (covers 512 B and 4 KiB sectors,
 * and the 16 KiB alignment the repacker uses for experts). */
#define KR_IO_ALIGN 65536u

typedef struct kr_io kr_io;

/* Called on an I/O thread when a read finishes. status is 0 on success, otherwise an OS error code. */
typedef void (*kr_io_done)(void * user, int status, size_t bytes_read);

enum kr_io_flags {
    KR_IO_DIRECT = 1, /* bypass the OS file cache (the default for weights and the speed test) */
};

/* Start an I/O engine with n_threads reader threads (the number of reads in flight). */
kr_io * kr_io_create(int n_threads);
void    kr_io_destroy(kr_io * io);   /* waits for all reads, then closes every file */

/* Open a file for reading. Returns a file id >= 0, or a negative error. */
int  kr_io_open(kr_io * io, const char * path, int flags);

/* Queue a read of len bytes at offset into dst. Returns 0 if queued. cb may be NULL. */
int  kr_io_read(kr_io * io, int file_id, uint64_t offset, size_t len, void * dst, kr_io_done cb, void * user);

/* Wait until every queued read has finished. */
void kr_io_wait(kr_io * io);

/* Totals since kr_io_create. */
uint64_t kr_io_bytes_read(const kr_io * io);

/* Memory aligned for direct reads. Free with kr_io_free. */
void * kr_io_alloc(size_t bytes);
void   kr_io_free(void * p);

/* Human-readable name of the OS method in use, e.g. "pread + O_DIRECT". */
const char * kr_io_backend_name(void);

#ifdef __cplusplus
}
#endif
