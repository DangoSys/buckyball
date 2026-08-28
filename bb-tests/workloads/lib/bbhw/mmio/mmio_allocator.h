#ifndef BBHW_MMIO_ALLOCATOR_H
#define BBHW_MMIO_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>

// Pebble MMIO is one 5 KiB byte-addressed space striped over five identical
// 8-bit x 1024 banks. A logical MMIO row is one normal 128-bit DMA beat.
#define MMIO_BANK_NUM 5
#define MMIO_BANK_ENTRIES 1024
#define MMIO_BANK_WIDTH_BITS 8
#define MMIO_BANK_BYTES ((MMIO_BANK_ENTRIES) * ((MMIO_BANK_WIDTH_BITS) / 8))
#define MMIO_TOTAL_BYTES ((MMIO_BANK_NUM) * (MMIO_BANK_BYTES))
#define MMIO_LOGICAL_ROW_BYTES 16
#define MMIO_LOGICAL_ROWS (MMIO_TOTAL_BYTES / MMIO_LOGICAL_ROW_BYTES)

// Bitmap-based allocator over the unified logical address space.
typedef struct {
  uint8_t bitmap[MMIO_LOGICAL_ROWS];
} mmio_allocator_t;

// Initialize allocator (all rows free).
void mmio_allocator_init(mmio_allocator_t *alloc);

// Allocate `size_rows` consecutive rows. Returns MMIO byte address, or
// (uint16_t)-1 on failure.
uint16_t mmio_allocator_alloc(mmio_allocator_t *alloc, uint16_t size_rows);

// Release rows previously returned by mmio_allocator_alloc.
void mmio_allocator_free(mmio_allocator_t *alloc, uint16_t addr,
                         uint16_t size_rows);

#endif
