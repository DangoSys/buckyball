#include <cstdio>
#include <kernel.h>
#include <rvv/launch.h>
#include <stream.h>

alignas(16) static const uint8_t image[] = {
#embed "rvv_stride.image"
};
alignas(16) static uint64_t input[64], zeros[64], output[5][64];
alignas(16) static operator_launch calls[5];
static constexpr int64_t input_steps[] = {1, 2, -1, 0, 3};
static constexpr int64_t output_steps[] = {2, 3, -2, 1, 1};
static void commands(void *) {
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)input, RVV_READ_BANK, 0, 32, 1);
  for (unsigned mode = 0; mode < 5; ++mode) {
    bb_mvin_group((uintptr_t)zeros, RVV_WRITE_BANK, 1, 32, 1);
    rvv_run(&calls[mode], sizeof(calls[mode]));
    bb_mvout_group((uintptr_t)output[mode], RVV_WRITE_BANK, 1, 32, 1);
    bb_mvout_group((uintptr_t)&calls[mode], RVV_WRITE_BANK, 0, 6, 1);
  }
  rvv_release();
}
int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  for (unsigned i = 0; i < 64; ++i)
    input[i] = UINT64_C(0xfeed000000000000) + i * UINT64_C(0x100020001);
  for (unsigned mode = 0; mode < 5; ++mode)
    calls[mode] = {rvv_image_word(image, 2),
                   rvv_image_word(image, 1),
                   RVV_STACK,
                   {(UINT64_C(3) << 32) + (output_steps[mode] < 0 ? 48 * 8 : 0),
                    input_steps[mode] < 0 ? UINT64_C(48) * 8 : 0,
                    (uint64_t)(input_steps[mode] * 8),
                    (uint64_t)(output_steps[mode] * 8), RVV_FLAGS, 0, 0, 0},
                   0};
  stream_run(slots[0].signature, commands, nullptr);
  for (unsigned mode = 0; mode < 5; ++mode) {
    uint64_t expected[64] = {};
    const auto length = calls[mode].reserved;
    if (length < 8 || length > 16)
      return 1;
    for (int64_t i = 0; i < (int64_t)length; ++i) {
      const auto src = (input_steps[mode] < 0 ? 48 : 0) + i * input_steps[mode];
      const auto dst =
          (output_steps[mode] < 0 ? 48 : 0) + i * output_steps[mode];
      expected[dst] = input[src];
    }
    for (unsigned i = 0; i < 64; ++i)
      if (output[mode][i] != expected[i]) {
        std::printf("RVV stride mismatch mode=%u element=%u\n", mode, i);
        return 1;
      }
  }
  std::puts("RVV STRIDE PASSED (positive, negative and zero input strides)");
  return 0;
}
