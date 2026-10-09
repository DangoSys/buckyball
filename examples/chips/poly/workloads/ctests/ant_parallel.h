#pragma once
#include <stdint.h>

struct ant_parallel_args {
  uint64_t shared, publish, input, output;
};

extern unsigned parallel_done[];
void parallel_exercise(unsigned tile);
