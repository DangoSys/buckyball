#include <bbhw/isa/isa.h>
#include <isa/smatmul.h>
#include <stdint.h>

static uint64_t cycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value)::"memory");
  return value;
}

uint64_t entry(uint64_t *args) {
  for (unsigned repeat = 0; repeat < 2; ++repeat) {
    uint64_t *timings = args + 4 + repeat * 7;
    uint64_t start = cycle();
    for (unsigned bank = 0; bank < 4; ++bank)
      bb_mem_alloc(bank, 1, 1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[0] = cycle() - start;
    start = cycle();
    bb_mvin(args[2], 0, 4, 1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[1] = cycle() - start;
    start = cycle();
    bb_mvin(args[0], 1, 16, 1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[2] = cycle() - start;
    start = cycle();
    bb_mvin(args[1], 2, 16, 1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[3] = cycle() - start;
    start = cycle();
    bb_smatmul_bias(0, 0);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[4] = cycle() - start;
    start = cycle();
    bb_smatmul_os(1, 2, 3, 16, 16, 16, 1, 1, 0);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[5] = cycle() - start;
    start = cycle();
    bb_mvout(args[3], 3, 64, 1);
    __asm__ volatile("fence rw, rw" ::: "memory");
    timings[6] = cycle() - start;
    for (unsigned bank = 0; bank < 4; ++bank)
      bb_mem_release(bank);
    __asm__ volatile("fence rw, rw" ::: "memory");
  }
  return 0;
}
