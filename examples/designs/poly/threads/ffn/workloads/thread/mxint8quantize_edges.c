// mxint8quantize on the thread alone: two rows of K = 64, two blocks each,
// with hand-picked block maxima: row 0 block 0 maximum 127.5 (exponent 0):
// ties to even and the clamp at 127, signed zeros; row 0 block 1 all zero
// (byte 127); row 1 block 0 maximum 254 (exponent 1); row 1 block 1 maximum
// 1 (exponent -6). The FP32 inputs are constants: the thread has no FPU.
#include <bbhw/isa/isa.h>
#include <isa/mxint8quantize.h>
#include <stdint.h>

enum {
  ROWS = 2,
  K = 64,
  STRIDE = MXINT8_ROWS(K) * 16,
  IN_BASE = 3,
  OUT_BASE = 2
};
#define INPUT ((volatile float *)HOST_TSS_BASE)
#define OUTPUT                                                                 \
  ((volatile uint8_t *)(HOST_TSS_BASE + (IN_BASE * 4 + ROWS * K) * 4))

static const float values[ROWS][8] = {
    {127.5f, -127.5f, 0.5f, 1.5f, 2.5f, -2.5f, 3.5f, -0.5f},
    {254.0f, 3.0f, 5.0f, -7.0f, 1.0f, 0.5f, 0.25f, 1e-3f}};
static const int8_t codes[ROWS][8] = {{127, -127, 0, 2, 2, -2, 4, 0},
                                      {127, 2, 2, -4, 64, 32, 16, 0}};
static const uint8_t exponents[ROWS][2] = {{127, 127}, {128, 121}};

uint64_t entry(void) {
  for (int i = 0; i < IN_BASE * 4 + ROWS * K; ++i)
    INPUT[i] = 0.0f;
  for (int i = 0; i < OUT_BASE * 16 + ROWS * STRIDE; ++i)
    OUTPUT[i] = 0;
  volatile float *x = INPUT + IN_BASE * 4;
  for (int i = 0; i < 6; ++i) {
    x[i] = values[0][i];
    x[K + i] = values[1][i % 4];
    x[K + 32 + i] = i < 4 ? values[1][4 + i] : 0.0f;
  }
  x[6] = values[0][6];
  x[7] = values[0][7];
  x[8] = -0.0f;
  x[9] = 126.49999f;

  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)INPUT, 1, IN_BASE + ROWS * K / 4, 1);
  bb_mxint8quant(1, 2, ROWS, K, IN_BASE, OUT_BASE);
  bb_mvout((uintptr_t)OUTPUT, 2, OUT_BASE + ROWS * MXINT8_ROWS(K), 1);
  asm volatile("fence" ::: "memory");

  for (int row = 0; row < ROWS; ++row) {
    volatile const uint8_t *packed = OUTPUT + OUT_BASE * 16 + row * STRIDE;
    for (int block = 0; block < 2; ++block)
      if (packed[K + block] != exponents[row][block])
        return 1 + row * 2 + block;
    for (int i = 0; i < K; ++i) {
      int expected = 0;
      if (row == 0 && i < 8)
        expected = codes[0][i];
      if (row == 0 && i == 9)
        expected = 126;
      if (row == 1 && i < 6)
        expected = codes[1][i % 4];
      if (row == 1 && i >= 32 && i < 36)
        expected = codes[1][4 + i - 32];
      if ((int8_t)packed[i] != expected)
        return 100 + row * K + i;
    }
    for (int i = K + 2; i < STRIDE; ++i)
      if (packed[i])
        return 300 + row * STRIDE + i;
  }
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
