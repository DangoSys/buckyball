#include <cstdio>
#include <kernel.h>
#include <rvv/launch.h>
#include <stream.h>

alignas(16) static const uint8_t image[] = {
#embed "rvv_gather.image"
};
alignas(16) static uint32_t input[32], indices[32], output[6][32];
alignas(16) static operator_launch calls[6];
static void commands(void *) {
  rvv_load(image, sizeof(image));
  bb_mvin_group((uintptr_t)input, RVV_READ_BANK, 0, 8, 1);
  bb_mvin_group((uintptr_t)indices, RVV_READ_BANK, 1, 8, 1);
  for (unsigned mode = 0; mode < 6; ++mode) {
    rvv_run(&calls[mode], sizeof(calls[mode]));
    bb_mvout_group((uintptr_t)output[mode], RVV_WRITE_BANK, 1, 8, 1);
    bb_mvout_group((uintptr_t)&calls[mode], RVV_WRITE_BANK, 0, 6, 1);
  }
  rvv_release();
}
int main() {
  kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  const uint32_t order[4] = {0, 16, 31, UINT32_MAX};
  for (unsigned i = 0; i < 32; ++i) {
    input[i] = UINT32_C(0x12345678) + i * UINT32_C(0x100001);
    indices[i] = order[i % 4];
  }
  for (unsigned mode = 0; mode < 6; ++mode)
    calls[mode] = {rvv_image_word(image, 2),
                   rvv_image_word(image, 1),
                   RVV_STACK,
                   {UINT64_C(3) << 32, 0, UINT64_C(1) << 32, mode, UINT64_MAX,
                    RVV_FLAGS, 0, 0},
                   0};
  stream_run(slots[0].signature, commands, nullptr);
  for (unsigned mode = 0; mode < 6; ++mode) {
    uint64_t length = calls[mode].reserved;
    if (length == 0 || length > 32)
      return 1;
    for (unsigned i = 0; i < length; ++i) {
      uint64_t selected = mode == 0 || mode == 5 ? 0
                          : mode == 1            ? 16
                          : mode == 2            ? 31
                          : mode == 3            ? UINT64_MAX
                                                 : indices[i];
      uint32_t expected = selected < length ? input[selected] : 0;
      if (mode == 5 && indices[i] != 0)
        expected = UINT32_C(0xdeadbeef);
      if (output[mode][i] != expected) {
        std::printf("RVV gather mismatch mode=%u lane=%u: %08x != %08x\n", mode,
                    i, output[mode][i], expected);
        return 1;
      }
    }
  }
  std::puts("RVV GATHER PASSED (unsigned immediates, VX, VV and masked lanes)");
  return 0;
}
