// Model file identity: a fast hash of the whole file (spec P3 bench results, and the KV store's cache
// identity in section 4b item 5).
#pragma once

#include <functional>
#include <string>

namespace kera {

// XXH3-128 of the whole file, as 32 hex digits (the same as `xxhsum -H2` prints). Reads with big
// direct reads, so hashing a model doesn't push other files out of memory. Returns "" on error,
// with the OS error code in *err. progress, if set, gets the fraction done.
std::string hash_file(const std::string & path, int * err = nullptr,
                      const std::function<void(double)> & progress = {});

// "xxh128:<hex>" for a model file. Remembered in the Kera folder by path, size and modification
// time, so each file is read once.
// known_only: only what's remembered, "" if the file would have to be read.
std::string model_hash(const std::string & path, int * err = nullptr, bool known_only = false);

// XXH3-128 of a block of memory, as 32 hex digits.
std::string hash_bytes(const void * data, size_t len);

} // namespace kera
