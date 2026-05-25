#include "mmio_allocator.h"

void mmio_allocator_init(mmio_allocator_t *alloc) {
  for (int i = 0; i < MMIO_BANK_NUM; i++) {
    alloc->bitmap[i] = 0;
  }
}

uint16_t mmio_allocator_alloc(mmio_allocator_t *alloc, uint16_t size_rows) {
  if (size_rows == 0 || size_rows > MMIO_BANK_ENTRIES) {
    return (uint16_t)-1;
  }
  for (int b = 0; b < MMIO_BANK_NUM; b++) {
    for (int start = 0; start + size_rows <= MMIO_BANK_ENTRIES; start++) {
      uint64_t mask;
      if (size_rows == 64) {
        mask = ~(uint64_t)0;
      } else {
        mask = (((uint64_t)1 << size_rows) - 1) << start;
      }
      if ((alloc->bitmap[b] & mask) == 0) {
        alloc->bitmap[b] |= mask;
        return (uint16_t)(MMIO_BANK_BASE(b) +
                          start * (MMIO_BANK_WIDTH_BITS / 8));
      }
    }
  }
  return (uint16_t)-1;
}

void mmio_allocator_free(mmio_allocator_t *alloc, uint16_t addr,
                         uint16_t size_rows) {
  if (size_rows == 0)
    return;
  int bytes_per_row = MMIO_BANK_WIDTH_BITS / 8;
  int b = addr / MMIO_BANK_BYTES;
  int start = (addr % MMIO_BANK_BYTES) / bytes_per_row;
  uint64_t mask;
  if (size_rows == 64) {
    mask = ~(uint64_t)0;
  } else {
    mask = (((uint64_t)1 << size_rows) - 1) << start;
  }
  alloc->bitmap[b] &= ~mask;
}
