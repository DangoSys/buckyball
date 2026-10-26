#pragma once
#include <stdint.h>
enum {
  CORELINK_TEST_BYTES = 4160,
  CORELINK_TEST_ROWS = CORELINK_TEST_BYTES / 16
};
struct ant_corelink_args {
  uint64_t phase, input, output;
};
