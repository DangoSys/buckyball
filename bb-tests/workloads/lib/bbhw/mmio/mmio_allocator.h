#ifndef BBHW_MMIO_ALLOCATOR_H
#define BBHW_MMIO_ALLOCATOR_H

#include <stddef.h>
#include <stdint.h>

// Default MMIO subsystem layout (must mirror MMIO config in hardware
// default.json):
//   16 banks x 64 entries x 128 bit = 16 KB total
#define MMIO_BANK_NUM 16
#define MMIO_BANK_ENTRIES 64
#define MMIO_BANK_WIDTH_BITS 128
#define MMIO_BANK_BYTES                                                        \
  ((MMIO_BANK_ENTRIES) * ((MMIO_BANK_WIDTH_BITS) / 8))         // 1024
#define MMIO_TOTAL_BYTES ((MMIO_BANK_NUM) * (MMIO_BANK_BYTES)) // 16384
#define MMIO_BANK_BASE(i) ((i) * (MMIO_BANK_BYTES))

// Bitmap-based allocator: one bit per row in each bank.
// 16 banks x 64 rows = 1024 bits = 16 uint64_t words.
typedef struct {
  uint64_t bitmap[MMIO_BANK_NUM]; // bitmap[i] bit j = row j of bank i used
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
