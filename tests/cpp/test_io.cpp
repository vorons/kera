// kr_io: parallel direct reads must return exactly the bytes in the file.

#include "kr_test.h"

#include "io/kr_io.h"
#include "io/kr_os_file.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::string temp_file(const char * name) {
#if defined(_WIN32)
    const char * dir = std::getenv("TEMP");
    return std::string(dir ? dir : ".") + "\\" + name;
#else
    return std::string("/tmp/") + name;
#endif
}

KR_TEST(parallel_direct_reads_return_the_file) {
    const std::string path = temp_file("kera_test_io.bin");
    const size_t block = KR_IO_ALIGN;
    const size_t n_blocks = 64;
    {
        kera::os::file f;
        KR_CHECK(kera::os::create_write(path, false, f) == 0);
        std::vector<uint8_t> buf(block);
        for (size_t b = 0; b < n_blocks; ++b) {
            for (size_t i = 0; i < block; ++i) {
                buf[i] = (uint8_t) ((b * 131 + i * 7) & 0xff);
            }
            size_t put = 0;
            KR_CHECK(kera::os::pwrite_full(f, b * block, block, buf.data(), &put) == 0);
            KR_CHECK(put == block);
        }
        kera::os::sync(f);
        kera::os::close(f);
    }

    kr_io * io = kr_io_create(8);
    const int fid = kr_io_open(io, path.c_str(), KR_IO_DIRECT);
    KR_CHECK(fid >= 0);

    std::vector<uint8_t *> dst(n_blocks);
    std::atomic<int> done{0};
    std::atomic<int> errors{0};
    struct ctx { std::atomic<int> * done; std::atomic<int> * errors; };
    ctx c{ &done, &errors };
    for (size_t b = 0; b < n_blocks; ++b) {
        dst[b] = (uint8_t *) kr_io_alloc(block);
        // read the blocks in reverse order, so nothing depends on sequential access
        const size_t src = n_blocks - 1 - b;
        KR_CHECK(kr_io_read(io, fid, src * block, block, dst[b], [](void * u, int status, size_t got) {
            ctx * c = (ctx *) u;
            if (status != 0 || got != KR_IO_ALIGN) (*c->errors)++;
            (*c->done)++;
        }, &c) == 0);
    }
    kr_io_wait(io);
    KR_CHECK(done == (int) n_blocks);
    KR_CHECK(errors == 0);
    KR_CHECK(kr_io_bytes_read(io) == n_blocks * block);

    bool same = true;
    for (size_t b = 0; b < n_blocks; ++b) {
        const size_t src = n_blocks - 1 - b;
        for (size_t i = 0; i < block; i += 997) {
            same = same && dst[b][i] == (uint8_t) ((src * 131 + i * 7) & 0xff);
        }
        kr_io_free(dst[b]);
    }
    KR_CHECK(same);
    kr_io_destroy(io);
    kera::os::remove_file(path);
}

KR_TEST(open_missing_file_fails) {
    kr_io * io = kr_io_create(1);
    KR_CHECK(kr_io_open(io, "/definitely/not/here.bin", KR_IO_DIRECT) < 0);
    kr_io_destroy(io);
}

KR_TEST_MAIN()
