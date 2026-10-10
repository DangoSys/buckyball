#include "buckyball.h"
#include <dma.h>
#include <isa/mxmm.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum { BYTES = BANK_LINES * BANK_WIDTH / 8 };
static uint8_t a[BYTES] __attribute__((aligned(64)));
static uint8_t packed[BYTES] __attribute__((aligned(64)));
static uint8_t w[BYTES] __attribute__((aligned(64)));
static uint8_t actual[BYTES] __attribute__((aligned(64)));
static uint8_t expected[BYTES] __attribute__((aligned(64)));

int main(void) {
  const int shapes[][4] = {{16, 32, 960, 480}, {1, 32, 4096, 0}};
  const uint8_t codes[] = {0,    0x80, 1,    0x81, 0x38,
                           0xb8, 0x37, 0xb7, 0x7e, 0xfe};
  const uint8_t scales[] = {0, 119, 127, 128};
  for (int test = 0; test < 2; ++test) {
    int m = shapes[test][0], n = shapes[test][1];
    int full = shapes[test][2], start = shapes[test][3];
    memset(a, 0x5a, sizeof(a));
    for (int row = 0; row < m; ++row) {
      for (int k = 0; k < full; ++k)
        a[row * full + k] = k < start ? 0x7f : codes[(k + row) % 10];
      for (int block = 0; block < full / 32; ++block)
        a[m * full + row * (full / 32) + block] =
            block < start / 32 ? 255 : scales[(block + row) % 4];
    }
    memset(actual, 0x5a, sizeof(actual));
    memset(expected, 0x5a, sizeof(expected));
    for (int bank = 3; bank <= 7; ++bank)
      bb_mem_alloc(bank, 1, 1);
    bb_mvin((uintptr_t)a, 3, BANK_LINES, 1);
    bb_mvin((uintptr_t)expected, 5, BANK_LINES, 1);
    bb_mvin((uintptr_t)actual, 7, BANK_LINES, 1);
    /* Complete old-71 chain before new-75 chain: one chain slot per hart. */
    for (int mode = 0; mode < 2; ++mode) {
      for (int k = start; k < full; k += 480) {
        int count = full - k < 480 ? full - k : 480;
        memset(w, 0x5a, sizeof(w));
        for (int col = 0; col < n; ++col) {
          for (int i = 0; i < count; ++i)
            w[col * count + i] = codes[(k + i + col + 4) % 10];
          for (int block = 0; block < count / 32; ++block)
            w[n * count + col * (count / 32) + block] =
                scales[(k / 32 + block + col) % 4];
        }
        if (mode == 0) {
          memset(packed, 0x5a, sizeof(packed));
          for (int row = 0; row < m; ++row) {
            memcpy(packed + row * count, a + row * full + k, count);
            memcpy(packed + m * count + row * (count / 32),
                   a + m * full + row * (full / 32) + k / 32, count / 32);
          }
        }
        bb_mvin((uintptr_t)w, 6, BANK_LINES, 1);
        if (mode == 0) {
          bb_mvin((uintptr_t)packed, 4, BANK_LINES, 1);
          bb_mxmm_mxfp8(4, 6, 5, m, n, count, k == start, k + count == full, 1);
          bb_mvout((uintptr_t)expected, 5, BANK_LINES, 1);
        } else {
          bb_mxmm_mxfp8_window(3, 6, 7, m, n, count, k == start,
                               k + count == full, 1, full, k);
          bb_mvout((uintptr_t)actual, 7, BANK_LINES, 1);
        }
        if (k + count != full)
          for (int i = 0; i < BYTES; ++i)
            if ((mode ? actual : expected)[i] != 0x5a)
              return 1;
      }
    }
    if (memcmp(actual, expected, BYTES))
      return 2;
    for (int i = 0; i < BYTES; ++i)
      if ((i < 16 || i >= 16 + m * n * 4) && actual[i] != 0x5a)
        return 3;
    for (int bank = 3; bank <= 7; ++bank)
      bb_mem_release(bank);
  }
  puts("mxmm A-window row-gap/4096-chain/scale-byte/guard PASS");
  return 0;
}
