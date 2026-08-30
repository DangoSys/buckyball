#include <buckyball/rushb.h>

#ifndef BUCKYBALL_RUSHB_ACCELERATOR_ID
#define BUCKYBALL_RUSHB_ACCELERATOR_ID 0
#endif

#ifndef BUCKYBALL_RUSHB_CHIP_ID
#define BUCKYBALL_RUSHB_CHIP_ID 0
#endif

static void __attribute__((constructor)) rushb_runtime_init(void) {
  rushb_init();
  rushb_select_accelerator(BUCKYBALL_RUSHB_ACCELERATOR_ID,
                           BUCKYBALL_RUSHB_CHIP_ID);
}

static void __attribute__((destructor)) rushb_runtime_destroy(void) {
  rushb_destroy();
}
