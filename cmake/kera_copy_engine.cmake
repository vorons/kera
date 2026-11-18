# Copies an engine's programs and shared libraries (from anywhere in its build folder FROM) into one folder TO,
# next to each other, as the downloads ship them: cmake -DFROM=... -DTO=... -P kera_copy_engine.cmake
file(MAKE_DIRECTORY "${TO}")
file(GLOB_RECURSE found LIST_DIRECTORIES false
     "${FROM}/ace-*" "${FROM}/trellis-*" "${FROM}/audiocpp_*" "${FROM}/*.so" "${FROM}/*.so.*" "${FROM}/*.dylib"
     "${FROM}/*.dll")
foreach(f IN LISTS found)
    get_filename_component(name "${f}" NAME)
    # the programs themselves, not CMake's own files or object files that happen to match
    if (name MATCHES "\\.(o|obj|cmake|txt|d|pdb|ilk|exp|lib)$" OR f MATCHES "/CMakeFiles/")
        continue()
    endif()
    file(COPY "${f}" DESTINATION "${TO}" FOLLOW_SYMLINK_CHAIN)
endforeach()
