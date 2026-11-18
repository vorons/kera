// Auto-setup step 1 (spec 4a): detect the chip, core counts, RAM, OS and drives, without admin rights.
// The graphics chip is detected through ggml's back-ends in measure_gpu.cpp.

#include "setup/kr_machine.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#include <sys/mount.h>
#include <sys/param.h>
#include <sys/sysctl.h>
#include <sys/statvfs.h>
#else
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/utsname.h>
#include <unistd.h>
#endif

namespace kera {

static constexpr double GB = 1e9;           // drives: decimal, the way drives are sold
static constexpr double RAM_GB = 1073741824.0; // memory: binary, the way memory is sold

static std::string trim(std::string s) {
    const char * ws = " \t\r\n";
    s.erase(0, s.find_first_not_of(ws));
    const size_t e = s.find_last_not_of(ws);
    s.erase(e == std::string::npos ? 0 : e + 1);
    return s;
}

#if defined(_WIN32)

static std::string narrow(const wchar_t * w, int n = -1) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, w, n, nullptr, 0, nullptr, nullptr);
    std::string s(len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, n, s.data(), len, nullptr, nullptr);
    if (n == -1 && !s.empty() && s.back() == '\0') {
        s.pop_back();
    }
    return s;
}

void detect_system(machine_profile & p) {
    p.os_family = "windows";

    wchar_t name[256] = {};
    DWORD size = sizeof(name);
    if (RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     L"ProcessorNameString", RRF_RT_REG_SZ, nullptr, name, &size) == ERROR_SUCCESS) {
        p.chip = trim(narrow(name));
    }

    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    std::vector<char> buf(len);
    if (len > 0 && GetLogicalProcessorInformationEx(RelationProcessorCore,
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
        int cores = 0;
        for (DWORD off = 0; off < len;) {
            auto * info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) (buf.data() + off);
            cores++;
            off += info->Size;
        }
        p.cpu_cores = cores;
    }
    p.cpu_threads = (int) GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);

    MEMORYSTATUSEX ms = {};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) {
        p.ram_gb = ms.ullTotalPhys / RAM_GB;
    }

    // GetVersionEx reports what the app's manifest claims; RtlGetVersion reports the truth
    using rtl_get_version_t = LONG (WINAPI *)(PRTL_OSVERSIONINFOW);
    auto rtl_get_version = (rtl_get_version_t) (void *) GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "RtlGetVersion");
    RTL_OSVERSIONINFOW vi = {};
    vi.dwOSVersionInfoSize = sizeof(vi);
    if (rtl_get_version && rtl_get_version(&vi) == 0) {
        const char * name_os = vi.dwMajorVersion >= 10 ? (vi.dwBuildNumber >= 22000 ? "Windows 11" : "Windows 10") : "Windows";
        p.os = std::string(name_os) + " (build " + std::to_string(vi.dwBuildNumber) + ")";
    } else {
        p.os = "Windows";
    }
}

std::vector<drive_profile> list_drives() {
    std::vector<drive_profile> res;
    const DWORD mask = GetLogicalDrives();
    for (int i = 0; i < 26; ++i) {
        if (!(mask & (1u << i))) {
            continue;
        }
        wchar_t root[] = { (wchar_t) (L'A' + i), L':', L'\\', 0 };
        if (GetDriveTypeW(root) != DRIVE_FIXED) {
            continue;
        }
        ULARGE_INTEGER avail = {}, total = {}, free_all = {};
        if (!GetDiskFreeSpaceExW(root, &avail, &total, &free_all)) {
            continue;
        }
        drive_profile d;
        d.path = narrow(root);
        d.total_gb = total.QuadPart / GB;
        d.free_gb = avail.QuadPart / GB;
        res.push_back(d);
    }
    return res;
}

double available_ram_gb() {
    MEMORYSTATUSEX m = {};
    m.dwLength = sizeof(m);
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys / RAM_GB : 0.0;
}

#elif defined(__APPLE__)

static std::string sysctl_string(const char * name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) {
        return std::string();
    }
    std::string s(len, '\0');
    if (sysctlbyname(name, s.data(), &len, nullptr, 0) != 0) {
        return std::string();
    }
    return trim(std::string(s.c_str()));
}

template <typename T>
static T sysctl_num(const char * name, T fallback) {
    T v = 0;
    size_t len = sizeof(v);
    return sysctlbyname(name, &v, &len, nullptr, 0) == 0 ? v : fallback;
}

void detect_system(machine_profile & p) {
    p.os_family = "macos";
    p.chip = sysctl_string("machdep.cpu.brand_string");
    p.cpu_cores = sysctl_num<int>("hw.physicalcpu", 0);
    p.cpu_threads = sysctl_num<int>("hw.logicalcpu", (int) std::thread::hardware_concurrency());
    p.ram_gb = sysctl_num<uint64_t>("hw.memsize", 0) / RAM_GB;
    const std::string ver = sysctl_string("kern.osproductversion");
    p.os = "macOS " + (ver.empty() ? std::string("(unknown version)") : ver);

    // Low Power Mode (spec 12.1 item 10): pmset reports it without admin rights
    if (FILE * f = popen("/usr/bin/pmset -g 2>/dev/null", "r")) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            std::string l = line;
            if (l.find("lowpowermode") != std::string::npos && trim(l).back() == '1') {
                p.low_power_mode = true;
            }
        }
        pclose(f);
    }
}

std::vector<drive_profile> list_drives() {
    std::vector<drive_profile> res;
    struct statfs * mnts = nullptr;
    const int n = getmntinfo(&mnts, MNT_NOWAIT);
    std::set<std::pair<int32_t, int32_t>> seen;
    for (int i = 0; i < n; ++i) {
        const struct statfs & m = mnts[i];
        const std::string mp = m.f_mntonname;
        if (!(m.f_flags & MNT_LOCAL) || (m.f_flags & MNT_RDONLY)) {
            continue; // the sealed system volume "/" is read-only; its data lives in /System/Volumes/Data
        }
        if (mp.rfind("/System/Volumes/", 0) == 0 && mp != "/System/Volumes/Data") {
            continue; // VM, Preboot, Update and other system helpers
        }
        if (mp.rfind("/dev", 0) == 0 || std::strcmp(m.f_fstypename, "devfs") == 0 || std::strcmp(m.f_fstypename, "autofs") == 0) {
            continue;
        }
        const auto key = std::make_pair(m.f_fsid.val[0], m.f_fsid.val[1]);
        if (!seen.insert(key).second) {
            continue;
        }
        drive_profile d;
        d.path = mp;
        d.device = m.f_mntfromname;
        d.total_gb = (double) m.f_blocks * m.f_bsize / GB;
        d.free_gb = (double) m.f_bavail * m.f_bsize / GB;
        res.push_back(d);
    }
    return res;
}

double available_ram_gb() {
    // free, inactive and purgeable pages: what the OS hands out without paging anything out
    vm_statistics64_data_t vm = {};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &n) != KERN_SUCCESS) {
        return 0.0;
    }
    const double page = (double) sysctl_num<uint64_t>("hw.pagesize", 16384);
    return ((double) vm.free_count + vm.inactive_count + vm.purgeable_count) * page / RAM_GB;
}

#else // Linux

static std::string read_text(const char * path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

double available_ram_gb() {
    std::istringstream m(read_text("/proc/meminfo"));
    std::string key;
    double kb = 0;
    std::string unit;
    while (m >> key >> kb >> unit) {
        if (key == "MemAvailable:") {
            return kb * 1024 / RAM_GB;
        }
    }
    return 0.0;
}

void detect_system(machine_profile & p) {
    p.os_family = "linux";

    std::istringstream cpuinfo(read_text("/proc/cpuinfo"));
    std::string line;
    std::set<std::pair<std::string, std::string>> cores;
    std::string phys_id = "0";
    while (std::getline(cpuinfo, line)) {
        const size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, colon));
        const std::string val = trim(line.substr(colon + 1));
        if ((key == "model name" || key == "Model") && p.chip.empty()) {
            p.chip = val;
        } else if (key == "physical id") {
            phys_id = val;
        } else if (key == "core id") {
            cores.insert({ phys_id, val });
        }
    }
    p.cpu_threads = (int) std::thread::hardware_concurrency();
    p.cpu_cores = cores.empty() ? p.cpu_threads : (int) cores.size();

    const long pages = sysconf(_SC_PHYS_PAGES);
    const long page = sysconf(_SC_PAGE_SIZE);
    p.ram_gb = (double) pages * (double) page / RAM_GB;

    std::string pretty;
    std::istringstream osr(read_text("/etc/os-release"));
    while (std::getline(osr, line)) {
        if (line.rfind("PRETTY_NAME=", 0) == 0) {
            pretty = line.substr(12);
            pretty.erase(std::remove(pretty.begin(), pretty.end(), '"'), pretty.end());
        }
    }
    struct utsname u;
    uname(&u);
    p.os = (pretty.empty() ? std::string("Linux") : pretty) + " (kernel " + u.release + ")";
    if (p.chip.empty()) {
        p.chip = u.machine;
    }
}

std::vector<drive_profile> list_drives() {
    static const std::set<std::string> fs_types = {
        "ext2", "ext3", "ext4", "xfs", "btrfs", "f2fs", "zfs", "ntfs", "ntfs3", "fuseblk", "exfat", "vfat", "bcachefs",
    };
    std::vector<drive_profile> res;
    std::map<dev_t, size_t> by_dev;
    std::istringstream mounts(read_text("/proc/self/mounts"));
    std::string dev, mp, type, opts;
    std::string rest;
    while (mounts >> dev >> mp >> type >> opts) {
        std::getline(mounts, rest);
        if (!fs_types.count(type) || dev.rfind("/dev/", 0) != 0 || dev.rfind("/dev/loop", 0) == 0) {
            continue;
        }
        if (opts.rfind("ro,", 0) == 0 || opts == "ro") {
            continue;
        }
        // /proc/mounts escapes spaces as \040
        for (size_t pos; (pos = mp.find("\\040")) != std::string::npos;) {
            mp.replace(pos, 4, " ");
        }
        struct stat st;
        struct statvfs vs;
        if (stat(mp.c_str(), &st) != 0 || statvfs(mp.c_str(), &vs) != 0) {
            continue;
        }
        drive_profile d;
        d.path = mp;
        d.device = dev;
        d.total_gb = (double) vs.f_blocks * vs.f_frsize / GB;
        d.free_gb = (double) vs.f_bavail * vs.f_frsize / GB;
        const auto it = by_dev.find(st.st_dev);
        if (it == by_dev.end()) {
            by_dev[st.st_dev] = res.size();
            res.push_back(d);
        } else if (mp.size() < res[it->second].path.size()) {
            res[it->second] = d; // the same file system mounted twice: keep the shorter path
        }
    }
    return res;
}

#endif

} // namespace kera
