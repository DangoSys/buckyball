#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static void fail(void) {
#ifdef BAREMETAL
  *(volatile uint32_t *)0x60000000 = 1;
  while (1) {
  }
#else
  exit(1);
#endif
}

#ifdef __cplusplus
extern "C"
#endif
    void check_result(float *w_alloc, float *w_align, int64_t w_off, int64_t wm,
                      int64_t wn, int64_t wsm, int64_t wsn, float *wt_alloc,
                      float *wt_align, int64_t wt_off, int64_t wtm, int64_t wtn,
                      int64_t wtsm, int64_t wtsn) {
  (void)w_alloc;
  (void)wt_alloc;
  (void)wsm;
  (void)wsn;
  (void)wtsm;
  (void)wtsn;
  if (wm != 80 || wn != 80 || wtm != 80 || wtn != 80)
    fail();
  for (int i = 0; i < 80; ++i) {
    for (int j = 0; j < 80; ++j) {
      float src = w_align[w_off + i * 80 + j];
      float dst = wt_align[wt_off + j * 80 + i];
      if (dst != src)
        fail();
    }
  }
  printf("PASSED: tile.tile_transpose 80x80 f32\n");
}
