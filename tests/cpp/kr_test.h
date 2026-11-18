// A minimal test harness: each test file is one executable run by CTest.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace kr_test {

struct test_case {
    const char * name;
    std::function<void()> fn;
};

inline std::vector<test_case> & registry() {
    static std::vector<test_case> r;
    return r;
}

inline int & failures() {
    static int f = 0;
    return f;
}

struct reg {
    reg(const char * name, std::function<void()> fn) { registry().push_back({ name, std::move(fn) }); }
};

inline int run_all() {
    for (const auto & t : registry()) {
        const int before = failures();
        t.fn();
        std::printf("%s %s\n", failures() == before ? "ok  " : "FAIL", t.name);
    }
    std::printf("%zu tests, %d failed checks\n", registry().size(), failures());
    return failures() == 0 ? 0 : 1;
}

} // namespace kr_test

#define KR_CAT2(a, b) a##b
#define KR_CAT(a, b) KR_CAT2(a, b)
#define KR_TEST(name)                                                               \
    static void name();                                                            \
    static kr_test::reg KR_CAT(kr_reg_, name)(#name, name);                        \
    static void name()

#define KR_CHECK(cond)                                                              \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::printf("  %s:%d: check failed: %s\n", __FILE__, __LINE__, #cond); \
            kr_test::failures()++;                                                 \
        }                                                                          \
    } while (0)

#define KR_CHECK_NEAR(a, b, tol)                                                                    \
    do {                                                                                           \
        const double kr_a = (a), kr_b = (b);                                                       \
        if (!(std::fabs(kr_a - kr_b) <= (tol))) {                                                  \
            std::printf("  %s:%d: %s = %.9g, expected %.9g (tol %g)\n", __FILE__, __LINE__, #a, kr_a, kr_b, (double) (tol)); \
            kr_test::failures()++;                                                                 \
        }                                                                                          \
    } while (0)

#define KR_TEST_MAIN() \
    int main() { return kr_test::run_all(); }
