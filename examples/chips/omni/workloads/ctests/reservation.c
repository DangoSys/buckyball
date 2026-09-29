#include <interconnect.h>
#include <stdio.h>

int main(void) {
  uint64_t chip = link_read(LINK_CHIP_ID);
  const uint64_t memory = link_read(LINK_MEMORY_BYTES);
  volatile uint64_t *source = (volatile uint64_t *)(0x80000000ULL + memory / 2);
  volatile uint64_t *target =
      (volatile uint64_t *)(0x80000000ULL + memory * 3 / 4);
  if (chip > 1)
    return 0;
  if (chip == 0) {
    uint64_t value, status;
    *target = 10;
    __asm__ volatile("lr.d %0, (%1)" : "=r"(value) : "r"(target) : "memory");
    __asm__ volatile("sc.d %0, %2, (%1)"
                     : "=r"(status)
                     : "r"(target), "r"(11ULL)
                     : "memory");
    if (value != 10 || status != 0 || *target != 11)
      return 1;
    __asm__ volatile("lr.d %0, (%1)" : "=r"(value) : "r"(target) : "memory");
    *source = 1;
    if (chip_send(1, (uintptr_t)source, (uintptr_t)target, 8, 1))
      return 2;
    while (!link_read(LINK_EVENT_COUNT)) {
    }
    if (link_read(LINK_EVENT_SOURCE) != 1)
      return 3;
    __asm__ volatile("sc.d %0, %2, (%1)"
                     : "=r"(status)
                     : "r"(target), "r"(99ULL)
                     : "memory");
    if (status == 0 || *target != 20)
      return 4;
    link_write(LINK_EVENT_ACK, 1);
    puts("cross-chip DMA invalidates LR/SC reservation PASSED");
  } else {
    while (!link_read(LINK_EVENT_COUNT)) {
    }
    link_write(LINK_EVENT_ACK, 1);
    *source = 20;
    if (chip_send(0, (uintptr_t)source, (uintptr_t)target, 8, 2))
      return 5;
  }
  return 0;
}
