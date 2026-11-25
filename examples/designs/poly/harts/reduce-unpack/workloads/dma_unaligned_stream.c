#include <bbhw/isa/isa.h>
#include <stdint.h>

enum {
  ROWS = THREAD_MEM_BANK_ENTRIES / 2,
  WORDS = ROWS * THREAD_MEM_BANK_WIDTH / 64
};
#define INPUT UINT64_C(0x0706050403020100)
#define SHIFTED UINT64_C(0x0302010007060504)
#define GUARD UINT64_C(0xa5a5a5a5a5a5a5a5)
static uint64_t input[WORDS + 2]
    __attribute__((aligned(4096))) = {[0 ... WORDS + 1] = INPUT};
static uint64_t output[WORDS + 16]
    __attribute__((aligned(4096))) = {[0 ... WORDS + 15] = GUARD};

int main(void) {
  input[0] ^= UINT64_C(1) << 63;
  input[WORDS] ^= 1;
  *(volatile uint64_t *)&output[8] = GUARD;
  *(volatile uint64_t *)&output[WORDS + 7] = GUARD;
  bb_mem_alloc(0, 1, 1);
  bb_mvin((uintptr_t)input + 4, 0, ROWS, 1);
  bb_mvout((uintptr_t)(output + 8), 0, ROWS, 1);
  // Bank diff checks every row after the DDR round trip.
  bb_mvin((uintptr_t)(output + 8), 0, ROWS, 1);
  bb_mem_release(0);
  if (output[8] != (SHIFTED ^ (UINT64_C(1) << 31)) ||
      output[WORDS + 7] != (SHIFTED ^ (UINT64_C(1) << 32)))
    return 1;
  for (unsigned i = 0; i < 8; ++i)
    if (output[i] != GUARD || output[WORDS + 8 + i] != GUARD)
      return 2;
  return 0;
}
