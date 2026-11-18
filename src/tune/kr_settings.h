/* Settings registry (spec P3, P1 item 2): every knob, so the auto-tuner can see and try it.
 *
 * Each new technique registers a setting with a safe default, off until its gate passes. Nothing is
 * hard-wired on, and the user never has to set anything by hand (section 4a).
 */
#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Register a setting. Registering the same name twice keeps the first registration and returns 0;
 * a new registration returns 1. def is clamped to [min, max]. */
int kr_setting_register(const char * name, double def, double min, double max, int tunable, const char * help);

/* Current value; def if never set, NaN if the setting doesn't exist. */
double kr_setting_get(const char * name);

/* Set a value (clamped to [min, max]). Returns 0 if the setting doesn't exist. */
int kr_setting_set(const char * name, double value);

/* Back to the default. */
void kr_setting_reset(const char * name);

typedef struct {
    const char * name;
    double       def;
    double       min;
    double       max;
    double       value;
    int          tunable;
    const char * help;
} kr_setting_info;

/* Number of registered settings, and the i-th one in registration order. */
size_t kr_settings_count(void);
int    kr_settings_get_info(size_t i, kr_setting_info * out);

/* Register the settings every part of Kera owns (called once at start-up). */
void kr_settings_register_builtin(void);

#ifdef __cplusplus
}
#endif
