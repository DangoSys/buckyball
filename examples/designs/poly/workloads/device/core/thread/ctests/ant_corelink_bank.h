#pragma once
#include <stdint.h>
enum {
  CORELINK_CROSS_BYTES = 192,
  CORELINK_CROSS_ROWS = CORELINK_CROSS_BYTES / 16
};
struct ant_corelink_bank_args {
  uint64_t phase, input, output;
};
