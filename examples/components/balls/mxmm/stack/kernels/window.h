#pragma once

#include <stdint.h>

struct Window {
  uint32_t phase;
  uint32_t width;
  uint32_t mean;
  uint32_t epsilon;
  uint32_t columns;
  uint32_t count;
  uint32_t full;
  uint32_t start;
  uint32_t first;
  uint32_t last;
  uint32_t rounding;
};
