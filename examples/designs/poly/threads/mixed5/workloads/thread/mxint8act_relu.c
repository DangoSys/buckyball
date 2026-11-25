// mxint8act on the thread alone: 2 packed MXINT8 rows of 64 through a ReLU
// table (Q26, 4096 entries) into another bank. The table fills the TSS, so
// it goes into its bank first and the TSS then holds the rows. The expected
// codes are the Ball's integer definition (emu/src/activation.rs).
#include <bbhw/isa/isa.h>
#include <isa/mxint8act.h>
#include <stdint.h>

enum { R = 2, K = 64, STRIDE = (K + K / 32 + 15) / 16 * 16 };
#define TABLE ((volatile int32_t *)HOST_TSS_BASE)
#define X ((volatile uint8_t *)HOST_TSS_BASE)
#define Y ((volatile uint8_t *)(HOST_TSS_BASE + R * STRIDE))

static int code(int r, int i) { return (r * 37 + i * 23) % 255 - 127; }
static int exponent(int r, int block) { return 127 - 3 - (r + block) % 4; }

static int64_t round_shift(int64_t value, int shift) {
  if (shift <= 0)
    return value * ((int64_t)1 << -shift);
  if (shift > 62)
    return 0;
  uint64_t magnitude = value < 0 ? -(uint64_t)value : (uint64_t)value;
  uint64_t quotient = magnitude >> shift;
  uint64_t remainder = magnitude & ((UINT64_C(1) << shift) - 1);
  uint64_t half = UINT64_C(1) << (shift - 1);
  quotient += remainder > half || (remainder == half && (quotient & 1));
  return value < 0 ? -(int64_t)quotient : (int64_t)quotient;
}

#define relu(index) ((index) < 2048 ? 0 : ((index) - 2048) << 19)

static int64_t lookup(int code, int e) {
  if (e < -24)
    return (int64_t)code * (1 << 15);
  int64_t x = (int64_t)code * ((int64_t)1 << ((e < 4 ? e : 4) + 26));
  if (x >= (int64_t)16 << 26)
    return (int64_t)code * (1 << 16);
  if (x <= -((int64_t)16 << 26))
    return 0;
  int64_t position = x + ((int64_t)16 << 26);
  int index = position >> 19;
  int64_t first = relu(index);
  int64_t last = index == 4095 ? (int64_t)16 << 26 : relu(index + 1);
  int64_t value =
      first + round_shift((last - first) * (position & ((1 << 19) - 1)), 19);
  return round_shift(value, e + 10);
}

uint64_t entry(void) {
  for (int i = 0; i < 4096; ++i)
    TABLE[i] = relu(i);
  for (int bank = 1; bank <= 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mvin((uintptr_t)TABLE, 2, 1024, 1);
  asm volatile("fence" ::: "memory");
  for (int r = 0; r < R; ++r) {
    for (int i = 0; i < STRIDE; ++i)
      X[r * STRIDE + i] = 0;
    for (int i = 0; i < K; ++i)
      X[r * STRIDE + i] = (uint8_t)code(r, i);
    for (int block = 0; block < K / 32; ++block)
      X[r * STRIDE + K + block] = exponent(r, block);
  }
  bb_mvin((uintptr_t)X, 1, R * STRIDE / 16, 1);
  bb_mxint8act(1, 2, 3, R, K, 0, 0, 0, 0);
  bb_mvout((uintptr_t)Y, 3, R * STRIDE / 16, 1);
  asm volatile("fence" ::: "memory");
  for (int r = 0; r < R; ++r)
    for (int block = 0; block < K / 32; ++block) {
      int e = exponent(r, block) - 127;
      int64_t values[32];
      uint64_t maximum = 0;
      for (int lane = 0; lane < 32; ++lane) {
        values[lane] = lookup(code(r, block * 32 + lane), e);
        uint64_t magnitude = values[lane] < 0 ? -values[lane] : values[lane];
        if (magnitude > maximum)
          maximum = magnitude;
      }
      int out = 0;
      if (maximum) {
        int msb = 63;
        while (!(maximum >> msb))
          --msb;
        out = e + msb - 22;
        out = out < -127 ? -127 : out > 127 ? 127 : out;
      }
      for (int lane = 0; lane < 32; ++lane) {
        int shift = out - e + 16;
        int64_t v = values[lane];
        int64_t c =
            shift < -6 ? 127 * ((v > 0) - (v < 0)) : round_shift(v, shift);
        c = c < -127 ? -127 : c > 127 ? 127 : c;
        if (Y[r * STRIDE + block * 32 + lane] != (uint8_t)(int8_t)c)
          return 1 + r * K + block * 32 + lane;
      }
      if (Y[r * STRIDE + K + block] != out + 127)
        return 1000 + r * 2 + block;
    }
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  return 0;
}
