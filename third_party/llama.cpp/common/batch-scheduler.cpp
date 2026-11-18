#include "batch-scheduler.h"

// [Kera] see batch-scheduler.h
static common_batch_scheduler * g_batch_scheduler = nullptr;

void common_set_batch_scheduler(common_batch_scheduler * s) {
    g_batch_scheduler = s;
}

common_batch_scheduler * common_get_batch_scheduler() {
    return g_batch_scheduler;
}
