#ifndef GOBAN_H
#define GOBAN_H

#include <stdint.h>
#include <bbhw/isa/isa.h>

/* Read hart ID from CSR mhartid */
static inline int bb_get_core_id(void) {
  int hartid;
  asm volatile("csrr %0, mhartid" : "=r"(hartid));
  return hartid;
}

#endif // GOBAN_H
