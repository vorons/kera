# Included into the engines' own builds (image, music, 3D, sound effects; CMAKE_PROJECT_INCLUDE): on Windows their
# programs get a manifest that makes UTF-8 their code page (Windows 10 1903 and later). They read their command line
# with plain main(argc, argv) and open files with fopen and FindFirstFileA, which otherwise use the old code page: a folder
# with a letter outside it (C:\Users\José\...) can't be opened, and kera passes them paths in UTF-8. (The sound-effect
# engine, audio.cpp, gives its programs the same manifest itself when built with MSVC's compiler; the downloads use
# clang's.)
get_property(kera_utf8_done GLOBAL PROPERTY KERA_UTF8_MANIFEST SET)
if ((WIN32 OR KERA_ENGINE_UTF8_TEST) AND NOT kera_utf8_done)
    # this runs after each project() of the build (ggml's too): once per configure is enough
    set_property(GLOBAL PROPERTY KERA_UTF8_MANIFEST "${CMAKE_CURRENT_LIST_DIR}/kera_utf8.manifest")
    function(kera_engine_utf8)
        get_property(manifest GLOBAL PROPERTY KERA_UTF8_MANIFEST)
        set(programs ace-lm ace-synth ace-server ace-understand sd-cli sd-server trellis-cli)
        if (NOT MSVC)
            list(APPEND programs audiocpp_cli) # audio.cpp adds its own, but only for MSVC's compiler
        endif()
        foreach(t ${programs})
            if (TARGET ${t})
                target_sources(${t} PRIVATE "${manifest}")
                message(STATUS "Kera: ${t} gets a UTF-8 code page")
            endif()
        endforeach()
    endfunction()
    # the programs are defined after project(): add the manifest once the whole build is read
    cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL kera_engine_utf8)
endif()
