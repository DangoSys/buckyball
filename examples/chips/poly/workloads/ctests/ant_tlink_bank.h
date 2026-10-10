#pragma once
#include <stdint.h>
enum { TLINK_CROSS_BYTES = 192, TLINK_CROSS_ROWS = TLINK_CROSS_BYTES / 16 };
struct ant_tlink_bank_args {
  uint64_t phase, input, output;
};
