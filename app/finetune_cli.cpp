// `kera finetune`: fine-tunes a model with a LoRA adapter (spec Phase 10, M11). Kera doesn't write a trainer:
// this runs tools/finetune/kera_finetune.py (MLX-LM on a Mac with Apple silicon, Hugging Face PEFT elsewhere) with
// the Python on this computer, and passes every option through. In a download the script sits in the
// `finetune` folder next to this program, with llama.cpp's converters.

#include "app/kera_app.h"
#include "common/kr_paths.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;

namespace kera {

namespace {

std::string find_script() {
    const fs::path exe = fs::u8path(exe_dir());
    for (const fs::path & p : { exe / "finetune" / "kera_finetune.py",                          // a download
                                exe / ".." / "Resources" / "finetune" / "kera_finetune.py",     // the Mac app
                                exe / ".." / ".." / "tools" / "finetune" / "kera_finetune.py",  // build/bin in the source tree
                                fs::u8path(KERA_SOURCE_DIR) / "tools" / "finetune" / "kera_finetune.py" }) { // a build elsewhere
        std::error_code ec;
        if (fs::is_regular_file(p, ec)) {
            return fs::weakly_canonical(p, ec).u8string();
        }
    }
    return "";
}

#if !defined(_WIN32)
// A command-line word, quoted for the shell std::system uses
std::string quote(const std::string & s) {
    std::string out = "'";
    for (const char c : s) {
        out += c == '\'' ? std::string("'\\''") : std::string(1, c);
    }
    return out + "'";
}
#endif

// the Python to run, as program words ("py", "-3"); empty if none
std::vector<std::string> find_python() {
    if (const char * p = std::getenv("KERA_PYTHON"); p != nullptr && *p != '\0') {
        return { p }; // a path, maybe with spaces ("C:\\Program Files\\...")
    }
#if defined(_WIN32)
    if (std::system("python --version > NUL 2>&1") == 0) {
        return { "python" };
    }
    if (std::system("py -3 --version > NUL 2>&1") == 0) {
        return { "py", "-3" };
    }
    return {};
#else
    return std::system("python3 --version > /dev/null 2>&1") == 0 ? std::vector<std::string>{ "python3" }
                                                                    : std::vector<std::string>{};
#endif
}

} // namespace

int run_finetune_cli(int argc, char ** argv) {
    const std::string script = find_script();
    if (script.empty()) {
        std::fprintf(stderr, "Can't find kera_finetune.py: it comes with Kera's download, in the finetune folder.\n");
        return 2;
    }
    const std::vector<std::string> python = find_python();
    if (python.empty()) {
        std::fprintf(stderr,
                     "Fine-tuning runs in Python, and none was found. Install Python 3.11 or newer (python.org), then\n"
                     "  pip install -r \"%s\"\n"
                     "(on a Mac with Apple silicon: pip install mlx-lm). Or set KERA_PYTHON to the Python to use.\n",
                     (fs::u8path(script).parent_path() / "requirements.txt").u8string().c_str());
        return 2;
    }
#if defined(_WIN32)
    // started directly (CreateProcessW), not through cmd.exe: cmd would expand %NAME% in the arguments and treat
    // & | > ^ after an escaped quote as its own operators
    std::vector<std::string> args = python;
    args.push_back(script);
    for (int i = 1; i < argc; ++i) {
        args.push_back(argv[i]);
    }
    const int rc = run_program(args);
#else
    std::string cmd;
    for (const auto & w : python) {
        cmd += (cmd.empty() ? "" : " ") + quote(w);
    }
    cmd += " " + quote(script);
    for (int i = 1; i < argc; ++i) {
        cmd += " " + quote(argv[i]);
    }
    const int rc_raw = std::system(cmd.c_str());
    const int rc = WIFEXITED(rc_raw) ? WEXITSTATUS(rc_raw) : 1;
#endif
    if (rc != 0 && argc <= 1) {
        std::fprintf(stderr, "\nIf Python said a package is missing: pip install -r \"%s\"\n",
                     (fs::u8path(script).parent_path() / "requirements.txt").u8string().c_str());
    }
    return rc;
}

} // namespace kera
