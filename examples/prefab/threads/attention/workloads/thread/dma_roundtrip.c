// The DMA paths of the thread alone: rows the thread writes into the TSS go
// TSS -> bank -> ideal memory -> bank -> TSS and come back unchanged.
#include <bbhw/isa/isa.h>
#include <stdint.h>

enum { ROWS = 16, BYTES = ROWS * 16 };
#define SOURCE ((volatile uint8_t *)HOST_TSS_BASE)
#define TARGET ((volatile uint8_t *)(HOST_TSS_BASE + BYTES))
#define MEMORY (HOST_MEMORY_BASE + 0x100000)

uint64_t entry(void) {
  for (int i = 0; i < BYTES; ++i) {
    SOURCE[i] = (uint8_t)(i * 7 + 3);
    TARGET[i] = 0;
  }
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin((uintptr_t)SOURCE, 0, ROWS, 1);
  bb_mvout(MEMORY, 0, ROWS, 1);
  bb_mvin(MEMORY, 1, ROWS, 1);
  bb_mvout((uintptr_t)TARGET, 1, ROWS, 1);
  asm volatile("fence" ::: "memory");
  for (int i = 0; i < BYTES; ++i)
    if (TARGET[i] != (uint8_t)(i * 7 + 3))
      return 1 + i;
  bb_mem_release(0);
  bb_mem_release(1);
  return 0;
}
