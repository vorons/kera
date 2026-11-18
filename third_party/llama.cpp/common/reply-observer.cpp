#include "reply-observer.h"

// [Kera] see reply-observer.h
static common_reply_observer * g_reply_observer = nullptr;

void common_set_reply_observer(common_reply_observer * obs) {
    g_reply_observer = obs;
}

common_reply_observer * common_get_reply_observer() {
    return g_reply_observer;
}
