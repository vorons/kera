#include "kr_test.h"

#include "tune/kr_settings.h"

#include <cmath>
#include <cstring>

KR_TEST(register_get_set_clamp) {
    KR_CHECK(kr_setting_register("test.knob", 4.0, 0.0, 8.0, 1, "a test knob") == 1);
    KR_CHECK(kr_setting_register("test.knob", 1.0, 0.0, 2.0, 0, "again") == 0); // first registration wins
    KR_CHECK_NEAR(kr_setting_get("test.knob"), 4.0, 0);
    KR_CHECK(kr_setting_set("test.knob", 100.0) == 1);
    KR_CHECK_NEAR(kr_setting_get("test.knob"), 8.0, 0); // clamped to max
    kr_setting_reset("test.knob");
    KR_CHECK_NEAR(kr_setting_get("test.knob"), 4.0, 0);
}

KR_TEST(unknown_setting_is_nan) {
    KR_CHECK(std::isnan(kr_setting_get("test.does_not_exist")));
    KR_CHECK(kr_setting_set("test.does_not_exist", 1.0) == 0);
}

KR_TEST(default_is_clamped_and_listed) {
    kr_setting_register("test.clamped", 50.0, 0.0, 10.0, 0, "default above max");
    KR_CHECK_NEAR(kr_setting_get("test.clamped"), 10.0, 0);
    bool found = false;
    for (size_t i = 0; i < kr_settings_count(); ++i) {
        kr_setting_info info;
        KR_CHECK(kr_settings_get_info(i, &info) == 1);
        if (std::strcmp(info.name, "test.clamped") == 0) {
            found = true;
            KR_CHECK(info.tunable == 0);
            KR_CHECK(std::strcmp(info.help, "default above max") == 0);
        }
    }
    KR_CHECK(found);
}

KR_TEST(builtin_settings_registered) {
    kr_settings_register_builtin();
    KR_CHECK(!std::isnan(kr_setting_get("setup.ssd_test_gb")));
    KR_CHECK(!std::isnan(kr_setting_get("io.reader_threads")));
}

KR_TEST_MAIN()
