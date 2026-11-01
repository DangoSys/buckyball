#ifndef _MEM_H_
#define _MEM_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes covered by mvin/mvout for the allocated bank groups. */
static inline size_t dma_span_bytes(uint64_t depth, uint64_t stride,
                                    uint32_t groups) {
  size_t row_bytes = (size_t)THREAD_MEM_BANK_WIDTH / 8;

  if (depth == 0)
    return 0;
  return ((size_t)(depth - 1) * (size_t)stride + 1) * row_bytes *
         (size_t)groups;
}

void dma_bank_set_cols(uint32_t bank_id, uint32_t cols);
void dma_bank_allocate(uint32_t bank_id, uint32_t cols);
uint32_t dma_bank_cols(uint32_t bank_id);
void dma_bank_transfer(uint64_t source, uint64_t target);

/*
 * Force private writable pages for a DMA buffer under Linux.
 * Untouched BSS is CoW-mapped to the shared zero page; accelerator stores
 * translate to that PPN and do not install a private page, so CPU reads stay 0.
 * Prefer clear_* or memset when the buffer should also be zeroed.
 */
void dma_touch(void *p, size_t n);

/* Touch host span for mvout using cols recorded by dma_bank_set_cols. */
void dma_touch_mvout_group(void *p, uint64_t depth, uint64_t stride);
void dma_touch_mvout(void *p, uint64_t depth, uint64_t stride,
                     uint32_t bank_id);

#ifdef __cplusplus
}
#endif

#endif // _MEM_H_
