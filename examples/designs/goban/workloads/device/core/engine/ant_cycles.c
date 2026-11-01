#include <stdint.h>

/* Each interval contains 16 operations and the final CSR read. */
#define MEASURE(slot, instruction)                                             \
  __asm__ volatile("csrr t1, mcycle\n.rept 16\n" instruction                   \
                   "\n.endr\ncsrr t2, mcycle\nsub %0, t2, t1"                  \
                   : "=r"(result[slot])                                        \
                   :                                                           \
                   : "t0", "t1", "t2", "memory")

uint64_t entry(uint64_t *result) {
  MEASURE(0, "addi t0, zero, 1");
  MEASURE(1, "ld t0, 0(tp)");
  MEASURE(2, "sd zero, 120(tp)");
  MEASURE(3, "mul t0, zero, zero");
  MEASURE(4, "divu t0, zero, sp");
  MEASURE(5, "divuw t0, zero, sp");
  MEASURE(6, "divu t0, sp, zero");
  MEASURE(7, "fence rw, rw");
  return 0;
}
