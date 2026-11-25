#pragma once
#include <stdint.h>

struct copy_task {
  uintptr_t input, output;
  uint64_t rows;
};
