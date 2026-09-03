#ifndef _MEM_H_
#define _MEM_H_

#include <params.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Bytes covered by mvin/mvout for a bank with `groups` (= alloc col, min 1). */
static inline size_t bb_dma_span_bytes(uint64_t depth, uint64_t stride,
                                       uint32_t groups) {
  size_t row_bytes = (size_t)BANK_WIDTH / 8;
  uint32_t g = groups < 1 ? 1 : groups;

  if (depth == 0)
    return 0;
  return ((size_t)(depth - 1) * (size_t)stride + 1) * row_bytes * (size_t)g;
}

void bb_dma_bank_set_cols(uint32_t bank_id, uint32_t cols);
uint32_t bb_dma_bank_cols(uint32_t bank_id);

/*
 * Force private writable pages for a DMA buffer under Linux.
 * Untouched BSS is CoW-mapped to the shared zero page; accelerator stores
 * translate to that PPN and do not install a private page, so CPU reads stay 0.
 * Prefer clear_* or memset when the buffer should also be zeroed.
 */
void bb_dma_touch(void *p, size_t n);

/* Touch host span for mvout using cols recorded by bb_mem_alloc. */
void bb_dma_touch_mvout(void *p, uint64_t depth, uint64_t stride,
                        uint32_t bank_id);

#ifdef __cplusplus
}
#endif

#endif // _MEM_H_
