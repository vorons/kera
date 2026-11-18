// Where Kera keeps its files: the machine profile, tuned settings, downloaded models and caches.
#pragma once

#include <string>

namespace kera {

// The Kera folder. KERA_HOME if set, otherwise:
//   Windows: %LOCALAPPDATA%\Kera   macOS: ~/Library/Application Support/Kera   Linux: ~/.local/share/kera
// Created if missing.
std::string home_dir();

// A path inside the Kera folder, e.g. home_path("kera_machine.json").
std::string home_path(const std::string & name);

// The folder the running executable is in.
std::string exe_dir();

// a/b with the OS separator.
std::string join(const std::string & a, const std::string & b);

char path_sep();

} // namespace kera
