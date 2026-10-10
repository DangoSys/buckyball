#pragma once
#include <stdint.h>
enum { ANT_BANK_ROWS = 16 };

struct ant_bank_rows_args {
  uint64_t input, private_out, shared_out, tail_out, shared_rows;
};

extern unsigned bank_rows_done[];
void bank_rows_exercise(unsigned tile);
