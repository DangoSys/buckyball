#pragma once
#include <stdint.h>
struct transpose_dma_args {
  uint64_t phase, input, output;
};
