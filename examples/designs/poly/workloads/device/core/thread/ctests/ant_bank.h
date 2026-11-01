#pragma once
#include <stdint.h>

struct ant_bank_args {
  uint64_t input, private_out, tail_out;
};

extern unsigned bank_done[];
void bank_exercise(unsigned core);
