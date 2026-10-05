// LR/SC retire on the first try for every width, alignment and ordering the
// kernel uses, including Linux's atomic_add_unless shape: lr.w / sc.w.rl on the
// upper word of a doubleword.
#include <stdint.h>

static volatile uint64_t line[8] __attribute__((aligned(64)));

#define LRSC(name, lr, sc, type)                                               \
  static int name(volatile type *p) {                                          \
    type old;                                                                  \
    uint64_t fail;                                                             \
    for (int tries = 0; tries < 16; tries++) {                                 \
      asm volatile(lr " %0, (%2)\n\t"                                          \
                      "addi %0, %0, 1\n\t" sc " %1, %0, (%2)"                  \
                   : "=&r"(old), "=&r"(fail)                                   \
                   : "r"(p)                                                    \
                   : "memory");                                                \
      if (!fail)                                                               \
        return 0;                                                              \
    }                                                                          \
    return 1;                                                                  \
  }

LRSC(word_rl, "lr.w", "sc.w.rl", uint32_t)
LRSC(word, "lr.w", "sc.w", uint32_t)
LRSC(word_aqrl, "lr.w.aq", "sc.w.aqrl", uint32_t)
LRSC(dword_rl, "lr.d", "sc.d.rl", uint64_t)

int main(void) {
  volatile uint32_t *low = (volatile uint32_t *)&line[2];
  volatile uint32_t *high = low + 1;
  line[2] = 0x0000000500000007ULL;
  int failed = 0;
  failed |= word_rl(high) << 0;
  failed |= word_rl(low) << 1;
  failed |= word(high) << 2;
  failed |= word_aqrl(high) << 3;
  failed |= dword_rl(&line[3]) << 4;
  if (failed)
    return failed;
  // Each successful SC added one to its own word only.
  if (line[2] != 0x0000000800000008ULL || line[3] != 1)
    return 0x20;
  return 0;
}
