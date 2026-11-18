// Model file hashes: the streamed whole-file hash must equal the one-shot hash of the same bytes,
// whatever the file size, and model_hash must remember it until the file changes.

#include "kr_test.h"

#include "common/kr_hash.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
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

static std::vector<unsigned char> pattern(size_t n) {
    std::vector<unsigned char> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = (unsigned char) ((i * 7 + 3) % 251);
    }
    return v;
}

static void write_file(const std::string & path, const std::vector<unsigned char> & data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    f.write((const char *) data.data(), (std::streamsize) data.size());
}

KR_TEST(known_answers) {
    // from Python's xxhash package (xxh3_128_hexdigest)
    KR_CHECK(kera::hash_bytes("", 0) == "99aa06d3014798d86001c324468d497f");
    KR_CHECK(kera::hash_bytes("abc", 3) == "06b05ab6733a618578af5f94892f3950");
    const auto p = pattern(1000003);
    KR_CHECK(kera::hash_bytes(p.data(), p.size()) == "df8c9e13fa75ffd5e8082f5864c32f3f");
}

KR_TEST(streamed_file_hash_matches_one_shot) {
    const std::string path = temp_file("kera_test_hash.bin");
    const size_t chunk = 16u << 20;
    for (size_t n : { (size_t) 0, (size_t) 1, (size_t) 1000003, chunk, 2 * chunk + 12345 }) {
        const auto data = pattern(n);
        write_file(path, data);
        int err = -1;
        const std::string h = kera::hash_file(path, &err);
        KR_CHECK(err == 0);
        KR_CHECK(h == kera::hash_bytes(data.data(), data.size()));
    }
    std::remove(path.c_str());
}

KR_TEST(model_hash_is_remembered_until_the_file_changes) {
    // a Kera folder of its own, not the real one
#if defined(_WIN32)
    _putenv_s("KERA_HOME", temp_file("kera_test_home").c_str());
#else
    setenv("KERA_HOME", temp_file("kera_test_home").c_str(), 1);
#endif
    const std::string path = temp_file("kera_test_model_hash.bin");
    auto data = pattern(4096);
    write_file(path, data);
    const std::string a = kera::model_hash(path);
    KR_CHECK(a == "xxh128:" + kera::hash_bytes(data.data(), data.size()));
    KR_CHECK(kera::model_hash(path) == a);
    data.push_back(1); // a different size always means a different file
    write_file(path, data);
    const std::string b = kera::model_hash(path);
    KR_CHECK(b == "xxh128:" + kera::hash_bytes(data.data(), data.size()));
    KR_CHECK(b != a);
    // the same size and the same date, other bytes (a model replaced in place, its date put back as rsync -t,
    // robocopy or touch -r do): read again, so saved chats never continue from another file's memory (N-018)
    {
        namespace fs = std::filesystem;
        const auto when = fs::last_write_time(fs::u8path(path));
        data[0] ^= 0xFF;
        write_file(path, data);
        fs::last_write_time(fs::u8path(path), when);
        const std::string c = kera::model_hash(path);
        KR_CHECK(c == "xxh128:" + kera::hash_bytes(data.data(), data.size()));
        KR_CHECK(c != b);
    }
    int err = 0;
    KR_CHECK(kera::model_hash(temp_file("kera_no_such_file.bin"), &err).empty());
    KR_CHECK(err != 0);
    std::remove(path.c_str());
}

KR_TEST_MAIN()
