#include <interconnect.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
  _Atomic uint64_t *owner = (_Atomic uint64_t *)CHIP_LINK_BUFFER;
  uint64_t zero = 0;
  if (!atomic_compare_exchange_strong(owner, &zero, 1))
    return 0;
  uint64_t chip = link_read(LINK_CHIP_ID), count = link_read(LINK_CHIP_COUNT);
  uint64_t physical = link_read(LINK_BUFFER_ADDRESS);
  if (count < 2 || link_read(LINK_BUFFER_BYTES) < 12288)
    return 1;
  volatile uint64_t *source = (volatile uint64_t *)(CHIP_LINK_BUFFER + 4096);
  volatile uint64_t *destination =
      (volatile uint64_t *)(CHIP_LINK_BUFFER + 8192);
  for (unsigned i = 0; i < 32; ++i)
    source[i] = chip * 100 + i;
  if (chip_send((chip + 1) % count, physical + 4096, physical + 8192, 256, 42))
    return 2;
  while (!link_read(LINK_EVENT_COUNT)) {
  }
  if (link_read(LINK_EVENT_SOURCE) != (chip + count - 1) % count ||
      link_read(LINK_EVENT_TAG) != 42 || link_read(LINK_EVENT_BYTES) != 256)
    return 3;
  __asm__ volatile("fence rw,rw" ::: "memory");
  for (unsigned i = 0; i < 32; ++i)
    if (destination[i] != ((chip + count - 1) % count) * 100 + i)
      return 4;
  link_write(LINK_EVENT_ACK, 1);
  printf("chip %lu: PK DDR INTERCONNECT PASSED\n", chip);
  return 0;
}
