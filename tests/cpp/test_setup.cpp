// Auto-setup: the profile survives a save and load, the explanation is plain words, and a quick real
// run on this machine produces sane numbers.

#include "kr_test.h"

#include "setup/kr_machine.h"

#include <cstdio>
#include <string>

KR_TEST(profile_round_trip) {
    kera::machine_profile p;
    p.kera_version = "0.1.0";
    p.measured_at = "2026-10-14T12:00:00Z";
    p.os = "Windows 11 (build 26100)";
    p.os_family = "windows";
    p.chip = "AMD Ryzen AI Max+ 395 w/ Radeon 8060S";
    p.cpu_cores = 16;
    p.cpu_threads = 32;
    p.ram_gb = 64;
    p.usable_gb = 50.2;
    kera::gpu_profile g;
    g.name = "AMD Radeon 8060S Graphics";
    g.backend = "Vulkan";
    g.max_alloc_gb = 32;
    g.unified = true;
    p.gpus.push_back(g);
    p.ram_gbps.gpu = 210;
    p.ram_gbps.cpu = 95.5;
    kera::drive_profile d;
    d.path = "C:\\";
    d.measured = true;
    d.read_gbps = 4.61;
    d.best_read_mib = 16;
    d.best_inflight = 16;
    p.drives.push_back(d);
    p.notes.push_back("a note");

    const auto j = p.to_json();
    const auto q = kera::machine_profile::from_json(j);
    KR_CHECK(q.to_json() == j);
    KR_CHECK(j["gpu"]["name"].get<std::string>() == "AMD Radeon 8060S Graphics"); // P3 shape
    KR_CHECK(j["drives"][0]["read_gbps"].get<double>() == 4.61);

    const std::string path = "kera_test_profile.json";
    KR_CHECK(kera::save_profile(p, path));
    kera::machine_profile r;
    KR_CHECK(kera::load_profile(r, path));
    KR_CHECK(r.to_json() == j);
    std::remove(path.c_str());
}

KR_TEST(explain_is_plain_words) {
    kera::machine_profile p;
    p.chip = "Apple M4";
    p.cpu_cores = 10;
    p.ram_gb = 16;
    p.usable_gb = 10;
    p.os = "macOS 26.0";
    p.ram_gbps.gpu = 110;
    p.low_power_mode = true;
    kera::drive_profile d;
    d.path = "/System/Volumes/Data";
    d.measured = true;
    d.read_gbps = 2.9;
    d.single_read_gbps = 1.6;
    d.best_read_mib = 4;
    d.best_inflight = 8;
    p.drives.push_back(d);
    const auto lines = kera::explain(p);
    std::string all;
    for (const auto & l : lines) all += l + "\n";
    KR_CHECK(all.find("2.9 GB/s") != std::string::npos);
    KR_CHECK(all.find("Low Power Mode") != std::string::npos);
    KR_CHECK(all.find("110 GB/s") != std::string::npos);
}

KR_TEST(explain_says_how_to_give_the_graphics_chip_more_memory) {
    kera::machine_profile p;
    p.chip = "AMD RYZEN AI MAX+ 395";
    p.ram_gb = 64;
    p.usable_gb = 50;
    p.os = "Windows 11";
    p.os_family = "windows";
    kera::gpu_profile g;
    g.name = "AMD Radeon 8060S Graphics";
    g.backend = "Vulkan";
    g.unified = true;
    g.total_gb = 16; // the part Windows sets aside for it
    p.gpus.push_back(g);
    std::string all;
    for (const auto & l : kera::explain(p)) all += l + "\n";
    KR_CHECK(all.find("Variable Graphics Memory") != std::string::npos);
    p.gpus[0].total_gb = 48; // already most of the memory: nothing to say
    all.clear();
    for (const auto & l : kera::explain(p)) all += l + "\n";
    KR_CHECK(all.find("Variable Graphics Memory") == std::string::npos);
}

KR_TEST(a_dgx_spark_shares_its_memory) {
    // llama.cpp's CUDA back-end reports the GB10 as a separate card; Kera knows it by name or by its memory
    KR_CHECK(kera::nvidia_shares_memory("CUDA", "NVIDIA GB10", 128.5, 119.5));
    KR_CHECK(kera::nvidia_shares_memory("CUDA", "NVIDIA Jetson AGX Thor", 128.0, 122.0));
#if defined(__linux__) && defined(__aarch64__)
    KR_CHECK(kera::nvidia_shares_memory("CUDA", "Some future NVIDIA chip", 120.0, 109.0)); // about all of the memory
    KR_CHECK(!kera::nvidia_shares_memory("CUDA", "NVIDIA H100", 96.0, 480.0));            // Grace Hopper: its own
#else
    KR_CHECK(!kera::nvidia_shares_memory("CUDA", "Some future NVIDIA chip", 120.0, 109.0)); // by name only here
#endif
    // a card with about as much memory as the PC (decimal GB against GiB): not shared
    KR_CHECK(!kera::nvidia_shares_memory("CUDA", "NVIDIA GeForce RTX 5090", 34.2, 31.3));
    KR_CHECK(!kera::nvidia_shares_memory("CUDA", "NVIDIA GeForce RTX 4090", 25.4, 31.6));
    KR_CHECK(!kera::nvidia_shares_memory("CUDA", "NVIDIA GeForce RTX 4090", 24.0, 64.0)); // a card of its own
    KR_CHECK(!kera::nvidia_shares_memory("Vulkan", "NVIDIA GB10", 128.5, 119.5)); // Vulkan says so itself
    KR_CHECK(!kera::nvidia_shares_memory("CUDA", "NVIDIA GeForce RTX 5090", 32.0, 0.0)); // memory unknown
}

KR_TEST(quick_setup_on_this_machine) {
    kera::setup_options o;
    o.quick = true;
    const auto p = kera::run_setup(o);
    KR_CHECK(p.ram_gb > 0.5);
    KR_CHECK(p.cpu_threads >= 1);
    KR_CHECK(p.ram_gbps.cpu > 0.1);
    KR_CHECK(p.cpu_q4_gbps > 0.01);
    KR_CHECK(!p.os.empty());
    KR_CHECK(!p.drives.empty());
    bool measured = false;
    for (const auto & d : p.drives) {
        measured = measured || d.measured;
        if (d.measured) {
            KR_CHECK(d.read_gbps > 0.0);
            KR_CHECK(d.best_inflight >= 1);
        }
    }
    KR_CHECK(measured);
    std::printf("  %s, %d cores, %.1f GB RAM, CPU reads %.1f GB/s, CPU 4-bit test %.1f GB/s\n",
                p.chip.c_str(), p.cpu_cores, p.ram_gb, p.ram_gbps.cpu, p.cpu_q4_gbps);
}

KR_TEST_MAIN()
