#include <bbhw/mem/mem.h>

#include <stdio.h>
#include <stdlib.h>

static uint8_t bank_cols[BANK_NUM];
static uint8_t bank_cols_init;

static void ensure_bank_cols(void) {
  uint32_t i;

  if (bank_cols_init)
    return;
  for (i = 0; i < BANK_NUM; i++)
    bank_cols[i] = 1;
  bank_cols_init = 1;
}

void bb_dma_bank_set_cols(uint32_t bank_id, uint32_t cols) {
  ensure_bank_cols();
  if (bank_id >= BANK_NUM) {
    fprintf(stderr, "bb_dma_bank_set_cols: bank_id %u out of range\n", bank_id);
    exit(1);
  }
  bank_cols[bank_id] = cols < 1 ? 1 : (cols > 255 ? 255 : (uint8_t)cols);
}

uint32_t bb_dma_bank_cols(uint32_t bank_id) {
  ensure_bank_cols();
  if (bank_id >= BANK_NUM) {
    fprintf(stderr, "bb_dma_bank_cols: bank_id %u out of range\n", bank_id);
    exit(1);
  }
  return bank_cols[bank_id];
}

void bb_dma_touch(void *p, size_t n) {
  volatile uint8_t *b = (volatile uint8_t *)p;
  size_t i;

  if (n == 0)
    return;
  for (i = 0; i < n; i += 4096)
    b[i] = b[i];
  b[n - 1] = b[n - 1];
}

void bb_dma_touch_mvout(void *p, uint64_t depth, uint64_t stride,
                        uint32_t bank_id) {
  bb_dma_touch(p, bb_dma_span_bytes(depth, stride, bb_dma_bank_cols(bank_id)));
}
