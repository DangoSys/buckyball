#pragma once
#include <stdint.h>
struct matmul_task {
  uint64_t a, b, bias, output;
};
