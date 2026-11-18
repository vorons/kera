cmake_minimum_required(VERSION 3.21)

project(kera VERSION 0.1.5.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(KERA_BUILD_TESTS          "Kera: build the unit tests"                      ON)
option(KERA_WARNINGS_AS_ERRORS   "Kera: treat warnings in Kera's own code as errors" OFF)
option(KERA_IMAGE                "Kera: build the image engine (stable-diffusion.cpp)" ON)
option(KERA_MUSIC                "Kera: build the music engine (acestep.cpp)" ON)
option(KERA_3D                   "Kera: build the 3D engine (trellis.cpp)" ON)
option(KERA_SFX                  "Kera: build the sound-effect engine (audio.cpp)" ON)

# Every executable and library lands in one folder, so the app finds its back-end libraries next to it.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# The downloads keep the program and its libraries side by side, so look for libraries next to the program.
if (APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path")
elseif (UNIX)
    set(CMAKE_BUILD_RPATH "\$ORIGIN")
    # with CUDA (the DGX Spark download), the CUDA libraries where NVIDIA installs them too, in case they aren't on the
    # system's library path: otherwise the CUDA part doesn't load and everything runs on the CPU
    set(KERA_CUDA_RPATH /usr/local/cuda/lib64 /usr/local/cuda/targets/sbsa-linux/lib /usr/local/cuda/targets/x86_64-linux/lib)
    if (GGML_CUDA)
        list(APPEND CMAKE_BUILD_RPATH ${KERA_CUDA_RPATH})
    endif()
endif()

# ---------------------------------------------------------------------------
# llama.cpp, copied into third_party/ with git subtree (docs/UPSTREAM_PATCHES.md)
# ---------------------------------------------------------------------------
set(LLAMA_BUILD_COMMON    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS     ON  CACHE BOOL "" FORCE) # the *-impl libraries that the kera binary links
set(LLAMA_BUILD_SERVER    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP       OFF CACHE BOOL "" FORCE) # Kera has its own binary
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI        OFF CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE) # no download at build time; Kera serves its own page
set(LLAMA_TOOLS_INSTALL   OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# llama.cpp's headers are someone else's code: Kera's warning flags (and -Werror) shouldn't apply to them
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
    foreach(t llama-common llama-common-base ggml llama mtmd cpp-httplib)
        if (TARGET ${t})
            set_property(TARGET ${t} PROPERTY SYSTEM TRUE)
        endif()
    endforeach()
endif()

# ---------------------------------------------------------------------------
# Kera
# ---------------------------------------------------------------------------
include(cmake/kera_version.cmake)
include(cmake/kera_licenses.cmake)

function(kera_target_warnings target)
    if (MSVC)
        target_compile_options(${target} PRIVATE /W4 /wd4100 /wd4996)
        if (KERA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wno-unused-parameter)
        if (KERA_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    endif()
    if (WIN32)
        # Windows' C library calls getenv, fopen and friends "unsafe"; llama.cpp silences this the same way
        target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
    endif()
endfunction()

add_subdirectory(src)
add_subdirectory(app)

# ggml's tiled matrix multiply on the CPU (llamafile's sgemm) for the image, music and 3D engines: on x86-64 only, where
# it was checked against the reference (D-030). On ARM its half-precision path adds in 8 lanes, about twice the rounding
# of stock ggml's 32 (where the CPU has SVE or int8 matrix maths it is compiled out anyway), so there the engines keep
# their own default, off (N-018)
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
    set(KERA_ENGINE_LLAMAFILE ON)
else()
    set(KERA_ENGINE_LLAMAFILE OFF)
endif()

# ---------------------------------------------------------------------------
# The image engine: stable-diffusion.cpp (third_party/stable-diffusion.cpp, docs/DECISIONS.md D-027), built as a
# project of its own with its own copy of ggml, so it never meets llama.cpp's: its programs and libraries land in
# bin/image/, and kera runs them as separate processes (kera image, and kera serve for an image model).
# ---------------------------------------------------------------------------
if (KERA_IMAGE)
    include(ExternalProject)
    set(KERA_IMAGE_OUT ${CMAKE_BINARY_DIR}/bin/image)
    set(KERA_IMAGE_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DSD_BUILD_SHARED_LIBS=ON
        -DSD_BUILD_SHARED_GGML_LIB=ON
        -DSD_WEBP=OFF
        -DSD_WEBM=OFF
        -DSD_SERVER_BUILD_FRONTEND=OFF
        -DGGML_NATIVE=${GGML_NATIVE}
        -DGGML_OPENMP=OFF
        # ggml's tiled matrix multiply on the CPU (llamafile's sgemm), which the chat engine builds with
        # (llama.cpp turns it on) and ggml alone leaves off. Pictures made on the CPU: 20.8 s instead of 38.5 s for
        # SD-Turbo, the VAE's decode 14.7 s instead of 29.8 s, and as close to the reference as before (D-030); on
        # x86-64 only (KERA_ENGINE_LLAMAFILE above)
        -DGGML_LLAMAFILE=${KERA_ENGINE_LLAMAFILE}
        -DCMAKE_PROJECT_INCLUDE=${CMAKE_SOURCE_DIR}/cmake/kera_engine_utf8.cmake)
    if (CMAKE_TOOLCHAIN_FILE)
        get_filename_component(kera_toolchain "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        list(APPEND KERA_IMAGE_ARGS -DCMAKE_TOOLCHAIN_FILE=${kera_toolchain})
    endif()
    # the downloads' CPU code: one copy per CPU family, the best one picked when it starts, as for the chat engine.
    # Without it an engine is built for AVX2 alone: it fails to start on CPUs without AVX2, and leaves AVX-512 unused
    # (the music model's diffusion step took 9.9 s instead of 5.8 s on an AVX-512 CPU; pictures were the same)
    if (GGML_BACKEND_DL)
        list(APPEND KERA_IMAGE_ARGS -DGGML_BACKEND_DL=ON)
    endif()
    if (GGML_CPU_ALL_VARIANTS)
        list(APPEND KERA_IMAGE_ARGS -DGGML_CPU_ALL_VARIANTS=ON)
    endif()
    if (CMAKE_CUDA_HOST_COMPILER)
        list(APPEND KERA_IMAGE_ARGS -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER})
    endif()
    if (CMAKE_C_COMPILER_LAUNCHER)
        list(APPEND KERA_IMAGE_ARGS -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()
    if (APPLE)
        list(APPEND KERA_IMAGE_ARGS -DSD_METAL=ON -DCMAKE_BUILD_RPATH=@loader_path
                                    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
    elseif (UNIX)
        if (GGML_CUDA)
            string(REPLACE ";" "|" kera_cuda_rpath "${KERA_CUDA_RPATH}")
            list(APPEND KERA_IMAGE_ARGS "-DCMAKE_BUILD_RPATH=\$ORIGIN|${kera_cuda_rpath}")
        else()
            list(APPEND KERA_IMAGE_ARGS "-DCMAKE_BUILD_RPATH=\$ORIGIN")
        endif()
    endif()
    if (GGML_VULKAN)
        list(APPEND KERA_IMAGE_ARGS -DSD_VULKAN=ON)
    endif()
    # NVIDIA (the DGX Spark download, D-029): the same chips as the chat engine
    if (GGML_CUDA)
        # without CUTLASS (an optional NVIDIA library for faster 8-bit maths, not copied in): to measure on a Spark
        list(APPEND KERA_IMAGE_ARGS -DSD_CUDA=ON "-DSD_CUDA_CUTLASS_PATH=")
        if (CMAKE_CUDA_ARCHITECTURES)
            list(APPEND KERA_IMAGE_ARGS "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
    # one configuration (bin/image, not bin/image/Release) whatever generator builds Kera
    set(kera_image_generator ${CMAKE_GENERATOR})
    if (CMAKE_GENERATOR MATCHES "Ninja")
        set(kera_image_generator Ninja)
    endif()
    ExternalProject_Add(kera-image
        LIST_SEPARATOR |
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/stable-diffusion.cpp
        BINARY_DIR ${CMAKE_BINARY_DIR}/image-engine
        CMAKE_GENERATOR ${kera_image_generator}
        CMAKE_ARGS ${KERA_IMAGE_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target sd-cli sd-server
        # its build puts everything in its own bin/: copied next to kera, into image/
        INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory <BINARY_DIR>/bin ${KERA_IMAGE_OUT}
        BUILD_ALWAYS ON
        USES_TERMINAL_BUILD ON)
    add_dependencies(kera kera-image)
endif()

# ---------------------------------------------------------------------------
# The music engine: acestep.cpp (third_party/acestep.cpp, docs/DECISIONS.md D-028), built the same way as the image
# engine: a project of its own with its own copy of ggml, its programs and libraries in bin/music/.
# ---------------------------------------------------------------------------
if (KERA_MUSIC)
    include(ExternalProject)
    set(KERA_MUSIC_ARGS
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DBUILD_SHARED_LIBS=ON
        -DGGML_NATIVE=${GGML_NATIVE}
        -DGGML_OPENMP=OFF
        # ggml's tiled matrix multiply on the CPU (llamafile's sgemm), which the chat engine builds with
        # (llama.cpp turns it on) and ggml alone leaves off. Pictures made on the CPU: 20.8 s instead of 38.5 s for
        # SD-Turbo, the VAE's decode 14.7 s instead of 29.8 s, and as close to the reference as before (D-030)
        -DGGML_LLAMAFILE=${KERA_ENGINE_LLAMAFILE}
        -DCMAKE_PROJECT_INCLUDE=${CMAKE_SOURCE_DIR}/cmake/kera_engine_utf8.cmake)
    if (CMAKE_TOOLCHAIN_FILE)
        get_filename_component(kera_toolchain "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        list(APPEND KERA_MUSIC_ARGS -DCMAKE_TOOLCHAIN_FILE=${kera_toolchain})
    endif()
    # the downloads' CPU code: one copy per CPU family, the best one picked when it starts, as for the chat engine.
    # Without it an engine is built for AVX2 alone: it fails to start on CPUs without AVX2, and leaves AVX-512 unused
    # (the music model's diffusion step took 9.9 s instead of 5.8 s on an AVX-512 CPU; pictures were the same)
    if (GGML_BACKEND_DL)
        list(APPEND KERA_MUSIC_ARGS -DGGML_BACKEND_DL=ON)
    endif()
    if (GGML_CPU_ALL_VARIANTS)
        list(APPEND KERA_MUSIC_ARGS -DGGML_CPU_ALL_VARIANTS=ON)
    endif()
    if (CMAKE_CUDA_HOST_COMPILER)
        list(APPEND KERA_MUSIC_ARGS -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER})
    endif()
    if (CMAKE_C_COMPILER_LAUNCHER)
        list(APPEND KERA_MUSIC_ARGS -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                                    -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()
    if (APPLE)
        list(APPEND KERA_MUSIC_ARGS -DGGML_METAL=ON -DCMAKE_BUILD_RPATH=@loader_path
                                    -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
    elseif (UNIX)
        if (GGML_CUDA)
            string(REPLACE ";" "|" kera_cuda_rpath "${KERA_CUDA_RPATH}")
            list(APPEND KERA_MUSIC_ARGS "-DCMAKE_BUILD_RPATH=\$ORIGIN|${kera_cuda_rpath}")
        else()
            list(APPEND KERA_MUSIC_ARGS "-DCMAKE_BUILD_RPATH=\$ORIGIN")
        endif()
    endif()
    if (GGML_VULKAN)
        list(APPEND KERA_MUSIC_ARGS -DGGML_VULKAN=ON)
    endif()
    if (GGML_CUDA)
        list(APPEND KERA_MUSIC_ARGS -DGGML_CUDA=ON)
        if (CMAKE_CUDA_ARCHITECTURES)
            list(APPEND KERA_MUSIC_ARGS "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
    set(kera_music_generator ${CMAKE_GENERATOR})
    if (CMAKE_GENERATOR MATCHES "Ninja")
        set(kera_music_generator Ninja)
    endif()
    ExternalProject_Add(kera-music
        LIST_SEPARATOR |
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/acestep.cpp
        BINARY_DIR ${CMAKE_BINARY_DIR}/music-engine
        CMAKE_GENERATOR ${kera_music_generator}
        CMAKE_ARGS ${KERA_MUSIC_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target ace-server ace-lm ace-synth
        # its programs land at the top of its build folder and ggml's libraries in ggml/src: both go to music/
        INSTALL_COMMAND ${CMAKE_COMMAND} -DFROM=<BINARY_DIR> -DTO=${CMAKE_BINARY_DIR}/bin/music
                        -P ${CMAKE_SOURCE_DIR}/cmake/kera_copy_engine.cmake
        BUILD_ALWAYS ON
        USES_TERMINAL_BUILD ON)
    add_dependencies(kera kera-music)
endif()

# What the 3D and sound-effect engines share with the two above: the compilers, ggml's CPU variants, the folder their
# libraries are looked for in, and the graphics back-end the chat engine uses (D-031)
function(kera_engine_args OUT)
    set(args
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DGGML_NATIVE=${GGML_NATIVE}
        -DGGML_OPENMP=OFF
        -DGGML_LLAMAFILE=${KERA_ENGINE_LLAMAFILE}
        -DCMAKE_PROJECT_INCLUDE=${CMAKE_SOURCE_DIR}/cmake/kera_engine_utf8.cmake)
    if (CMAKE_TOOLCHAIN_FILE)
        get_filename_component(kera_toolchain "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE BASE_DIR "${CMAKE_SOURCE_DIR}")
        list(APPEND args -DCMAKE_TOOLCHAIN_FILE=${kera_toolchain})
    endif()
    if (GGML_BACKEND_DL)
        list(APPEND args -DGGML_BACKEND_DL=ON)
    endif()
    if (GGML_CPU_ALL_VARIANTS)
        list(APPEND args -DGGML_CPU_ALL_VARIANTS=ON)
    endif()
    if (CMAKE_CUDA_HOST_COMPILER)
        list(APPEND args -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER})
    endif()
    if (CMAKE_C_COMPILER_LAUNCHER)
        list(APPEND args -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
                         -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER})
    endif()
    if (APPLE)
        list(APPEND args -DGGML_METAL=ON -DCMAKE_BUILD_RPATH=@loader_path
                         -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET})
    elseif (UNIX)
        if (GGML_CUDA)
            string(REPLACE ";" "|" kera_cuda_rpath "${KERA_CUDA_RPATH}")
            list(APPEND args "-DCMAKE_BUILD_RPATH=\$ORIGIN|${kera_cuda_rpath}")
        else()
            list(APPEND args "-DCMAKE_BUILD_RPATH=\$ORIGIN")
        endif()
    endif()
    if (GGML_VULKAN)
        list(APPEND args -DGGML_VULKAN=ON)
    endif()
    if (GGML_CUDA)
        list(APPEND args -DGGML_CUDA=ON)
        if (CMAKE_CUDA_ARCHITECTURES)
            list(APPEND args "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
    set(${OUT} ${args} PARENT_SCOPE)
endfunction()
# one configuration (bin/3d, not bin/3d/Release) whatever generator builds Kera
set(kera_engine_generator ${CMAKE_GENERATOR})
if (CMAKE_GENERATOR MATCHES "Ninja")
    set(kera_engine_generator Ninja)
endif()

# ---------------------------------------------------------------------------
# The 3D engine: trellis.cpp (third_party/trellis.cpp, docs/DECISIONS.md D-031), Microsoft's TRELLIS.2 on its own copy
# of ggml, built like the others: its programs and libraries in bin/3d/. Without its optional quad retopology (CGAL,
# GPL) and WebP textures (a download at build time; the textures are PNG).
# ---------------------------------------------------------------------------
if (KERA_3D)
    include(ExternalProject)
    kera_engine_args(KERA_3D_ARGS)
    list(APPEND KERA_3D_ARGS -DBUILD_SHARED_LIBS=ON -DTRELLIS_WEBP=OFF)
    # trellis-cli links Vulkan's loader itself, for its own two kernels: on Windows it is loaded when first used, so the
    # program starts (its --help, the download's checks) on a PC without one; kera 3d says what is missing there
    if (WIN32 AND GGML_VULKAN)
        if (MSVC)
            list(APPEND KERA_3D_ARGS "-DCMAKE_EXE_LINKER_FLAGS=/DELAYLOAD:vulkan-1.dll delayimp.lib")
        else()
            list(APPEND KERA_3D_ARGS "-DCMAKE_EXE_LINKER_FLAGS=-Wl,/DELAYLOAD:vulkan-1.dll -ldelayimp")
        endif()
    endif()
    ExternalProject_Add(kera-3d
        LIST_SEPARATOR |
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/trellis.cpp
        BINARY_DIR ${CMAKE_BINARY_DIR}/3d-engine
        CMAKE_GENERATOR ${kera_engine_generator}
        CMAKE_ARGS ${KERA_3D_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target trellis-cli
        # its programs and ggml's libraries land at the top of its build folder: both go to 3d/
        INSTALL_COMMAND ${CMAKE_COMMAND} -DFROM=<BINARY_DIR> -DTO=${CMAKE_BINARY_DIR}/bin/3d
                        -P ${CMAKE_SOURCE_DIR}/cmake/kera_copy_engine.cmake
        BUILD_ALWAYS ON
        USES_TERMINAL_BUILD ON)
    add_dependencies(kera kera-3d)
endif()

# ---------------------------------------------------------------------------
# The sound-effect engine: audio.cpp (third_party/audio.cpp, docs/DECISIONS.md D-031), with only its sound-effect
# models (MOSS-SoundEffect v2, Stable Audio), built like the others: its program and libraries in bin/sfx/.
# ---------------------------------------------------------------------------
if (KERA_SFX)
    include(ExternalProject)
    kera_engine_args(KERA_SFX_ARGS)
    # audio.cpp sets ggml's options from its own: the CPU variants (which are shared libraries) and native code
    if (GGML_NATIVE)
        list(APPEND KERA_SFX_ARGS -DENGINE_ENABLE_NATIVE_CPU=ON)
    else()
        list(APPEND KERA_SFX_ARGS -DENGINE_ENABLE_NATIVE_CPU=OFF)
    endif()
    if (GGML_CPU_ALL_VARIANTS)
        list(APPEND KERA_SFX_ARGS -DENGINE_ENABLE_CPU_ALL_VARIANTS=ON)
    endif()
    # its models' specifications built in (none to install next to it), its own threads without OpenMP (which
    # neither Apple's compiler nor the downloads' other folders carry), and only the sound-effect models
    list(APPEND KERA_SFX_ARGS -DAUDIOCPP_DEPLOYMENT_BUILD=ON -DENGINE_ENABLE_OPENMP=OFF
                              -DAUDIOCPP_MODEL_SET=custom "-DAUDIOCPP_MODELS=moss_soundeffect,stable_audio")
    ExternalProject_Add(kera-sfx
        LIST_SEPARATOR |
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/third_party/audio.cpp
        BINARY_DIR ${CMAKE_BINARY_DIR}/sfx-engine
        CMAKE_GENERATOR ${kera_engine_generator}
        CMAKE_ARGS ${KERA_SFX_ARGS}
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release --target audiocpp_cli audiocpp_server
        # its build puts the program and the libraries in its own bin/: copied next to kera, into sfx/
        INSTALL_COMMAND ${CMAKE_COMMAND} -DFROM=<BINARY_DIR>/bin -DTO=${CMAKE_BINARY_DIR}/bin/sfx
                        -P ${CMAKE_SOURCE_DIR}/cmake/kera_copy_engine.cmake
        BUILD_ALWAYS ON
        USES_TERMINAL_BUILD ON)
    add_dependencies(kera kera-sfx)
endif()

if (KERA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
