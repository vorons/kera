# Writes kera_version.h with the version and the git commit the build came from.

find_package(Git QUIET)
set(KERA_GIT_COMMIT "unknown")
if (KERA_COMMIT)
    # given by the builder: a source snapshot has no git history to ask (packaging/linux/build-spark.sh)
    set(KERA_GIT_COMMIT "${KERA_COMMIT}")
elseif (GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=12 HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE KERA_GIT_COMMIT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if (NOT KERA_GIT_COMMIT)
        set(KERA_GIT_COMMIT "unknown")
    endif()
endif()

# The pinned llama.cpp release, from docs/UPSTREAM_PATCHES.md
set(KERA_LLAMA_CPP_TAG "b12500")

configure_file(${CMAKE_SOURCE_DIR}/cmake/kera_version.h.in ${CMAKE_BINARY_DIR}/generated/kera_version.h @ONLY)
