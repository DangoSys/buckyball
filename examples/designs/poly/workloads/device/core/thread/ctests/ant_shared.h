#pragma once
#include <stdint.h>

struct ant_shared_args {
  uint64_t shared, producer, value;
};

extern unsigned shared_done[];
void shared_exercise(unsigned core);
