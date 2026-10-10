#pragma once
#include <stdint.h>
enum { TLINK_TEST_BYTES = 4160, TLINK_TEST_ROWS = TLINK_TEST_BYTES / 16 };
struct ant_tlink_args {
  uint64_t phase, input, output;
};
