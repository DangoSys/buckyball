#include <cstdio>
#include <cstring>
#include <kernel.h>
#include <rvv/launch.h>
#include <stream.h>

alignas(16) static const uint8_t image[] = {
#embed "rvv_multiply.image"
};
alignas(16) static uint8_t left[4][32], right[4][32], output[4][8][32];
alignas(16) static operator_launch calls[4][8];
static constexpr uint64_t seed = UINT64_C(0x92345678fedcba91);
static void commands(void *) {
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)left, RVV_READ_BANK, 0, 8, 1);
  bb_mvin_group((uintptr_t)right, RVV_READ_BANK, 1, 8, 1);
  for (unsigned sew = 0; sew < 4; ++sew)
    for (unsigned mode = 0; mode < 8; ++mode) {
      rvv_run(&calls[sew][mode], sizeof(operator_launch));
      bb_mvout_group((uintptr_t)output[sew][mode], RVV_WRITE_BANK, 1, 2, 1);
    }
  rvv_release();
}
int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (unsigned sew = 0; sew < 4; ++sew) {
    unsigned bits = 8u << sew;
    uint64_t mask = UINT64_MAX >> (64 - bits), sign = UINT64_C(1) << (bits - 1);
    const uint64_t a[4] = {sign, mask, sign - 1, seed & mask};
    const uint64_t b[4] = {mask, sign, 2, (seed >> 7) & mask};
    for (unsigned lane = 0; lane < 4; ++lane) {
      std::memcpy(&left[sew][lane * (bits / 8)], &a[lane], bits / 8);
      std::memcpy(&right[sew][lane * (bits / 8)], &b[lane], bits / 8);
    }
    for (unsigned mode = 0; mode < 8; ++mode)
      calls[sew][mode] = {rvv_image_word(image, 2),
                          rvv_image_word(image, 1),
                          RVV_STACK,
                          {UINT64_C(3) << 32, sew * 32,
                           (UINT64_C(1) << 32) + sew * 32, mode, sew, seed, 0,
                           0},
                          0};
  }
  stream_run(slots[0].signature, commands, nullptr);
  for (unsigned sew = 0; sew < 4; ++sew) {
    unsigned bits = 8u << sew;
    uint64_t mask = UINT64_MAX >> (64 - bits);
    for (unsigned mode = 0; mode < 8; ++mode)
      for (unsigned lane = 0; lane < 4; ++lane) {
        uint64_t a = 0, b = 0, got = 0;
        std::memcpy(&a, &left[sew][lane * (bits / 8)], bits / 8);
        std::memcpy(&b, &right[sew][lane * (bits / 8)], bits / 8);
        std::memcpy(&got, &output[sew][mode][lane * (bits / 8)], bits / 8);
        int64_t sa = (int64_t)(a << (64 - bits)) >> (64 - bits);
        int64_t sb = (int64_t)(b << (64 - bits)) >> (64 - bits);
        uint64_t c = (a ^ seed) & mask, expected;
        switch (mode) {
        case 0:
          expected = a * b;
          break;
        case 1:
          expected = (unsigned __int128)a * b >> bits;
          break;
        case 2:
          expected = (__int128)sa * b >> bits;
          break;
        case 3:
          expected = (__int128)sa * sb >> bits;
          break;
        case 4:
          expected = b * c + a;
          break;
        case 5:
          expected = a - b * c;
          break;
        case 6:
          expected = a * b + c;
          break;
        case 7:
          expected = c - a * b;
          break;
        default:
          __builtin_trap();
        }
        if (got != (expected & mask)) {
          std::printf("RVV multiply mismatch sew=%u mode=%u lane=%u: %016lx != "
                      "%016lx\n",
                      bits, mode, lane, got, expected & mask);
          return 1;
        }
      }
  }
  std::puts("RVV MULTIPLY PASSED (e8/e16/e32/e64, low/high/mixed products and "
            "accumulates)");
  return 0;
}
