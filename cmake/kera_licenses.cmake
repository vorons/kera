# THIRD_PARTY_LICENSES.txt: the licence of every third-party part built into Kera's downloads, which
# their licences require the downloads to carry. `kera licenses` prints it, and each download ships it.
#
# llama.cpp collects some of its own (llama.cpp, cpp-httplib, and LLVM OpenMP when the Windows build
# fetches it) in the global LICENSE_TEXT property; this adds the rest of what it bundles.

set(KERA_LLAMA_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)
set(KERA_THIRD_PARTY_LICENSES ${CMAKE_BINARY_DIR}/generated/THIRD_PARTY_LICENSES.txt)

function(kera_license_section OUT NAME TEXT)
    set(title "${NAME}")
    string(REGEX REPLACE "." "=" underline "${title}")
    set(${OUT} "${${OUT}}${title}\n${underline}\n\n${TEXT}\n\n" PARENT_SCOPE)
endfunction()

function(kera_license_file OUT NAME FILE)
    if (NOT EXISTS "${FILE}")
        message(FATAL_ERROR "Licence file not found: ${FILE}")
    endif()
    file(READ "${FILE}" text)
    string(STRIP "${text}" text)
    kera_license_section(${OUT} "${NAME}" "${text}")
    set(${OUT} "${${OUT}}" PARENT_SCOPE)
endfunction()

set(all "Third-party software in Kera\n===========================\n\n")
string(APPEND all "Kera's downloads include the following third-party software. Each part keeps its own\n")
string(APPEND all "licence, reproduced below. Kera's own licence is in LICENSE, and NOTICE lists the parts.\n\n\n")

# what llama.cpp registered (stored as C++ raw strings for its own binary: turn them back into text)
get_property(llama_text GLOBAL PROPERTY LICENSE_TEXT)
string(REPLACE "R\"=L=(" "" llama_text "${llama_text}")
string(REPLACE ")=L=\",\n" "\n\n" llama_text "${llama_text}")
string(APPEND all "${llama_text}")

kera_license_file(all "License for nlohmann/json (JSON for Modern C++)" ${KERA_LLAMA_DIR}/licenses/LICENSE-jsonhpp)
kera_license_file(all "License for xxHash" ${KERA_LLAMA_DIR}/vendor/hash/xxhash/LICENSE)
kera_license_file(all "License for rotate-bits" ${KERA_LLAMA_DIR}/vendor/hash/rotate-bits/LICENSE.md)
kera_license_section(all "License for SHA-256 (from the LZMA SDK, by Igor Pavlov)" "Public domain.")
kera_license_section(all "License for SHA-1 (by Steve Reid)" "Public domain.")
kera_license_section(all "License for miniaudio (by David Reid)" "Public domain or MIT No Attribution, at the user's choice.")
kera_license_section(all "License for stb_image (by Sean Barrett)" "Public domain or MIT, at the user's choice; used here under the public-domain option.")
kera_license_section(all "License for subprocess.h (by Neil Henning)" "The Unlicense: free and unencumbered software released into the public domain.")
if (GGML_VULKAN)
    kera_license_section(all "License for the Vulkan headers (The Khronos Group)" [=[Copyright 2015-2025 The Khronos Group Inc.
SPDX-License-Identifier: Apache-2.0 OR MIT; used here under the MIT licence:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.]=])
endif()

# the image engine (D-027): stable-diffusion.cpp, its own copy of ggml, and what it bundles
if (KERA_IMAGE)
    set(KERA_SD_DIR ${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp)
    kera_license_file(all "License for stable-diffusion.cpp (the image engine)" ${KERA_SD_DIR}/LICENSE)
    kera_license_file(all "License for ggml (the image engine's copy)" ${KERA_SD_DIR}/ggml/LICENSE)
    kera_license_file(all "License for Oniguruma" ${KERA_SD_DIR}/thirdparty/oniguruma/COPYING)
    kera_license_file(all "License for utf8proc" ${KERA_SD_DIR}/thirdparty/utf8proc/LICENSE.md)
    kera_license_file(all "License for darts-clone" ${KERA_SD_DIR}/thirdparty/LICENSE.darts_clone.txt)
    kera_license_section(all "License for miniz (by Rich Geldreich)" "MIT: see the notice at the start of miniz.h in stable-diffusion.cpp's sources.")
    kera_license_section(all "License for zip (by Kuba Podgórski)" "The Unlicense: free and unencumbered software released into the public domain.")
    kera_license_section(all "License for stb_image_write and stb_image_resize (by Sean Barrett)" "Public domain or MIT, at the user's choice; used here under the public-domain option.")
endif()

# the music engine (D-028): acestep.cpp, its own copy of ggml, and what it bundles
if (KERA_MUSIC)
    set(KERA_ACE_DIR ${CMAKE_SOURCE_DIR}/third_party/acestep.cpp)
    kera_license_file(all "License for acestep.cpp (the music engine, with its MP3 encoder and its page)" ${KERA_ACE_DIR}/LICENSE)
    kera_license_file(all "License for ggml (the music engine's copy)" ${KERA_ACE_DIR}/ggml/LICENSE)
    kera_license_file(all "License for cpp-httplib (the music engine's copy)" ${KERA_ACE_DIR}/vendor/cpp-httplib/LICENSE)
    kera_license_section(all "License for yyjson (Copyright (c) 2020 YaoYuan)" "MIT: see the notice at the start of yyjson.h in acestep.cpp's sources.")
    kera_license_section(all "License for minimp3 (by lieff)" "CC0 1.0: dedicated to the public domain.")
endif()

# the 3D engine (D-031): trellis.cpp, its own copy of ggml, and what it bundles
if (KERA_3D)
    set(KERA_TRELLIS_DIR ${CMAKE_SOURCE_DIR}/third_party/trellis.cpp)
    kera_license_file(all "License for trellis.cpp (the 3D engine)" ${KERA_TRELLIS_DIR}/LICENSE)
    kera_license_file(all "License for ggml (the 3D engine's copy)" ${KERA_TRELLIS_DIR}/thirdparty/ggml/LICENSE)
    kera_license_file(all "License for meshoptimizer" ${KERA_TRELLIS_DIR}/thirdparty/meshoptimizer/LICENSE.md)
    kera_license_file(all "License for Fast-Quadric-Mesh-Simplification" ${KERA_TRELLIS_DIR}/thirdparty/fqms/LICENSE.md)
    kera_license_section(all "License for xatlas (Copyright (c) 2018-2020 Jonathan Young; Thekla, Inc; NVIDIA Corporation)" "MIT: see the notice at the start of xatlas.cpp in trellis.cpp's sources.")
    kera_license_section(all "License for stb_image, stb_image_write and stb_image_resize (the 3D engine's copies, by Sean Barrett)" "Public domain or MIT, at the user's choice; used here under the public-domain option.")
endif()

# the sound-effect engine (D-031): audio.cpp, its own copy of ggml, and what it bundles
if (KERA_SFX)
    set(KERA_AUDIOCPP_DIR ${CMAKE_SOURCE_DIR}/third_party/audio.cpp)
    kera_license_file(all "License for audio.cpp (the sound-effect engine)" ${KERA_AUDIOCPP_DIR}/LICENSE)
    kera_license_file(all "License for ggml (the sound-effect engine's copy)" ${KERA_AUDIOCPP_DIR}/external/ggml/LICENSE)
    kera_license_file(all "License for SentencePiece" ${KERA_AUDIOCPP_DIR}/external/sentencepiece/LICENSE)
    kera_license_file(all "License for Abseil (in SentencePiece)" ${KERA_AUDIOCPP_DIR}/external/sentencepiece/third_party/absl/LICENSE)
    kera_license_file(all "License for protobuf-lite (in SentencePiece)" ${KERA_AUDIOCPP_DIR}/external/sentencepiece/third_party/protobuf-lite/LICENSE)
    kera_license_file(all "License for darts-clone (in SentencePiece)" ${KERA_AUDIOCPP_DIR}/external/sentencepiece/third_party/darts_clone/LICENSE)
    kera_license_file(all "License for esaxx (in SentencePiece)" ${KERA_AUDIOCPP_DIR}/external/sentencepiece/third_party/esaxx/LICENSE)
    kera_license_file(all "License for cJSON" ${KERA_AUDIOCPP_DIR}/external/cJSON/LICENSE)
    kera_license_file(all "License for libyaml" ${KERA_AUDIOCPP_DIR}/external/libyaml/License)
    kera_license_file(all "License for utf8proc (the sound-effect engine's copy)" ${KERA_AUDIOCPP_DIR}/external/utf8proc/LICENSE.md)
    kera_license_section(all "License for the llama.cpp tokenizer in audio.cpp (Copyright (c) 2023-2026 The ggml authors)" "MIT, as llama.cpp's licence above.")
endif()

# write only when it changed, so builds don't redo the embedding for nothing
set(old "")
if (EXISTS ${KERA_THIRD_PARTY_LICENSES})
    file(READ ${KERA_THIRD_PARTY_LICENSES} old)
endif()
if (NOT old STREQUAL all)
    file(WRITE ${KERA_THIRD_PARTY_LICENSES} "${all}")
endif()
