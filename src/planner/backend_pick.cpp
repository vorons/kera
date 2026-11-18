// The back-end choice (backend_pick.h).

#include "planner/backend_pick.h"

#include <algorithm>
#include <cmath>

namespace kera {

double backend_speed::read_seconds(double held, double n) const {
    return n * (read_s + read_s_per_depth * (held + n / 2));
}

double backend_speed::write_seconds(double held, double g) const {
    return g * (write_s + write_s_per_depth * (held + g / 2));
}

double backend_speed::reply_seconds(double held, double read, double written) const {
    return read_seconds(held, read) + write_seconds(held + read, written);
}

namespace {

// time per token = a + b * x at two points: (x1, y1), (x2, y2); b is never negative
void fit_line(double x1, double y1, double x2, double y2, double & a, double & b) {
    if (std::fabs(x2 - x1) < 1e-9) {
        a = (y1 + y2) / 2;
        b = 0;
        return;
    }
    b = (y2 - y1) / (x2 - x1);
    if (b < 0) {
        b = 0;
        a = (y1 + y2) / 2;
        return;
    }
    a = y1 - b * x1;
    if (a <= 0) { // a steep slope from noise: keep the first point's speed for an empty chat
        a = y1;
        b = 0;
    }
}

} // namespace

backend_speed fit_backend_speed(const std::string & backend,
                                double read_d1, double read_n1, double read_t1,
                                double read_d2, double read_n2, double read_t2,
                                double write_d1, double write_g1, double write_t1,
                                double write_d2, double write_g2, double write_t2) {
    backend_speed s;
    s.backend = backend;
    if (read_n1 <= 0 || read_n2 <= 0 || write_g1 <= 0 || write_g2 <= 0) {
        return s;
    }
    // each measurement is a time per token at its middle depth
    fit_line(read_d1 + read_n1 / 2, read_t1 / read_n1, read_d2 + read_n2 / 2, read_t2 / read_n2, s.read_s, s.read_s_per_depth);
    fit_line(write_d1 + write_g1 / 2, write_t1 / write_g1, write_d2 + write_g2 / 2, write_t2 / write_g2, s.write_s,
             s.write_s_per_depth);
    return s;
}

std::vector<reply_shape> default_replies(int n_ctx) {
    std::vector<reply_shape> r;
    r.push_back({ 1500, 200, 300, 9 });                                       // chat messages
    const double doc = std::min(65536.0, std::max(0.0, n_ctx / 2.0 - 512));
    if (doc >= 1024) {
        r.push_back({ 0, doc, 300, 1 });                                      // now and then a long document
    }
    return r;
}

backend_pick pick_backend(const std::vector<backend_speed> & measured, const std::vector<reply_shape> & replies) {
    struct scored {
        std::string backend;
        double seconds;
    };
    std::vector<scored> all;
    double total_weight = 0;
    for (const auto & r : replies) {
        total_weight += r.weight;
    }
    for (const auto & m : measured) {
        if (!m.ok()) {
            continue;
        }
        double t = 0;
        for (const auto & r : replies) {
            t += r.weight * m.reply_seconds(r.held, r.read, r.written);
        }
        all.push_back({ m.backend, total_weight > 0 ? t / total_weight : 0 });
    }
    std::sort(all.begin(), all.end(), [](const scored & a, const scored & b) { return a.seconds < b.seconds; });
    backend_pick p;
    if (!all.empty()) {
        p.backend = all[0].backend;
        p.seconds = all[0].seconds;
    }
    if (all.size() > 1) {
        p.other = all[1].backend;
        p.other_seconds = all[1].seconds;
    }
    return p;
}

} // namespace kera
