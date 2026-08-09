#include <buckyball/rushb.h>

static void __attribute__((constructor)) rushb_runtime_init(void) {
  rushb_init();
}

static void __attribute__((destructor)) rushb_runtime_destroy(void) {
  rushb_destroy();
}
