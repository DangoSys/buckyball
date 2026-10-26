#pragma once
#include <stdint.h>

enum { ANT_M = 16, ANT_N = 16, ANT_K = 32 };
struct ant_mxmm_args {
  uint64_t a;
  uint64_t b;
  uint64_t output;
};
