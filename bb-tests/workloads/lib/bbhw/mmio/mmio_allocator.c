#include "mmio_allocator.h"

void mmio_allocator_init(mmio_allocator_t *alloc) {
  for (int i = 0; i < MMIO_LOGICAL_ROWS; i++) {
    alloc->bitmap[i] = 0;
  }
}

uint16_t mmio_allocator_alloc(mmio_allocator_t *alloc, uint16_t size_rows) {
  if (size_rows == 0 || size_rows > MMIO_LOGICAL_ROWS) {
    return (uint16_t)-1;
  }
  for (int start = 0; start + size_rows <= MMIO_LOGICAL_ROWS; start++) {
    int free = 1;
    for (int row = 0; row < size_rows; row++) {
      if (alloc->bitmap[start + row]) {
        free = 0;
        break;
      }
    }
    if (free) {
      for (int row = 0; row < size_rows; row++) {
        alloc->bitmap[start + row] = 1;
      }
      return (uint16_t)(start * MMIO_LOGICAL_ROW_BYTES);
    }
  }
  return (uint16_t)-1;
}

void mmio_allocator_free(mmio_allocator_t *alloc, uint16_t addr,
                         uint16_t size_rows) {
  if (size_rows == 0)
    return;
  int start = addr / MMIO_LOGICAL_ROW_BYTES;
  if (start + size_rows > MMIO_LOGICAL_ROWS) {
    return;
  }
  for (int row = 0; row < size_rows; row++) {
    alloc->bitmap[start + row] = 0;
  }
}
