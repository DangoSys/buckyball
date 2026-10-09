#pragma once

#include <stdint.h>

struct PackMM {
  uint32_t fullK;
  uint32_t fullN;
  uint32_t startK;
  uint32_t startN;
  uint32_t countK;
  uint32_t columns;
  uint32_t rows;
  uint32_t first;
  uint32_t last;
  uint32_t fused;
  uint32_t rounding;
};
