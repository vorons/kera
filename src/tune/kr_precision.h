// A model file's precision, from its name, and the same model at a higher precision next to it (D-036).
//
// A model's own weights are its full-precision file (BF16, F16, F32); 8-bit and 4-bit files are smaller copies that
// lose a little of it. Where Kera chooses among a folder's files it takes the highest precision that fits; where the
// user names a file, Kera says when a higher-precision copy of the same model is beside it.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kera {

// 0: 32-bit; 1: 16-bit (BF16, F16, FP16); 2: 8-bit; 3: 6-bit; 4: 5-bit; 5: 4-bit; 6 and up: lower; -1: the name has no
// precision tag. stem: the name in lower case without the tag and the extension, the same for every precision of a
// model ("flux-3-image-klein" for "flux-3-image-klein-BF16.gguf" and "FLUX-3-image-klein-Q8_0.gguf").
int precision_of(const std::string & file_name, std::string * stem = nullptr);

struct sibling_file {
    std::string name;
    uint64_t    bytes = 0;
};

// Of the files beside `name` (the same folder's listing), the same model at the highest precision above `name`'s that
// is at most fit_bytes large (0: any size); "" if there is none. Only files of the same kind (extension) count.
std::string higher_precision_sibling(const std::string & name, const std::vector<sibling_file> & folder,
                                     uint64_t fit_bytes);

} // namespace kera
