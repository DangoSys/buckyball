#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  volatile uint32_t *sim_exit = (volatile uint32_t *)0x60000000;
  *sim_exit = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

extern "C" void check_result(int8_t *allocated, int8_t *aligned, int64_t offset,
                             int64_t size0, int64_t size1, int64_t stride0,
                             int64_t stride1, int8_t *selectedAllocated,
                             int8_t *selectedAligned, int64_t selectedOffset,
                             int64_t selectedSize0, int64_t selectedSize1,
                             int64_t selectedStride0, int64_t selectedStride1) {
  (void)allocated;
  (void)selectedAllocated;
  if (size0 != 16 || size1 != 48 || stride0 != 48 || stride1 != 1 ||
      selectedSize0 != 16 || selectedSize1 != 32 || selectedStride0 != 32 ||
      selectedStride1 != 1) {
    fail();
  }
  int8_t *out = aligned + offset;
  int8_t *selected = selectedAligned + selectedOffset;
  for (int i = 0; i < 16; ++i) {
    for (int j = 0; j < 48; ++j) {
      int expected = j >= 16 && j < 32 ? i * 32 + j - 16 + 7 : i * 48 + j;
      if (out[i * stride0 + j] != (int8_t)expected) {
        fail();
      }
    }
    for (int j = 0; j < 32; ++j) {
      int expected = j < 16 ? i * 32 + j + 7 : 91;
      if (selected[i * selectedStride0 + j] != (int8_t)expected) {
        fail();
      }
    }
  }
  printf("PASSED: selected DMA preserves other groups and stride gaps\n");
}
