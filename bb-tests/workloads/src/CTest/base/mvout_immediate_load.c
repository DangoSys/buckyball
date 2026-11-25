#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>

enum { ROWS = 16, WORDS = ROWS * THREAD_MEM_BANK_WIDTH / 32, LAST = WORDS - 1 };
static const uint32_t source[WORDS]
    __attribute__((aligned(64))) = {[LAST] = 0x12345678};
static volatile uint32_t fenced[WORDS]
    __attribute__((aligned(64))) = {[LAST] = 0x11223344};
static volatile uint32_t direct[WORDS]
    __attribute__((aligned(64))) = {[LAST] = 0x11223344};

int main(void) {
  bb_mem_alloc(0, 1, 1);
  bb_mvin((uintptr_t)source, 0, ROWS, 1);
  asm volatile("fence rw,rw" ::: "memory");

  // Warm the last word of each transfer. Keep each instruction sequence in one
  // cache line.
  if (fenced[LAST] != 0x11223344 || direct[LAST] != 0x11223344)
    return 1;
  uint32_t after_fence, immediate;
  const uint64_t bank = BB_BANK0(0) | BB_ITER((uint64_t)ROWS);
  const uint64_t stride = FIELD(1ULL, 39, 57);
  asm volatile(".balign 64\n\t"
               ".insn r " STR(CUSTOM_3) ", 3, %c1, x0, %2, %3\n\t"
                                        "fence rw,rw\n\t"
                                        "lw %0, 0(%4)"
               : "=r"(after_fence)
               : "i"(BB_MVOUT_FUNC7), "r"(bank),
                 "r"(FIELD((uintptr_t)fenced, 0, 38) | stride),
                 "r"(&fenced[LAST])
               : "memory");
  asm volatile(".balign 64\n\t"
               ".insn r " STR(CUSTOM_3) ", 3, %c1, x0, %2, %3\n\t"
                                        "lw %0, 0(%4)"
               : "=r"(immediate)
               : "i"(BB_MVOUT_FUNC7), "r"(bank),
                 "r"(FIELD((uintptr_t)direct, 0, 38) | stride),
                 "r"(&direct[LAST])
               : "memory");
  asm volatile("fence rw,rw" ::: "memory");
  uint32_t final = direct[LAST];
  bb_mem_release(0);
  printf("fenced=%08x immediate=%08x final=%08x expected=%08x\n", after_fence,
         immediate, final, source[LAST]);
  return after_fence != source[LAST] || immediate != source[LAST] ||
         final != source[LAST];
}
