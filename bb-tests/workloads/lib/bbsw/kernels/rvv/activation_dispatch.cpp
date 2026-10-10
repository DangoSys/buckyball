#include "images.h"
#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <params.h>

namespace {
constexpr unsigned readBank = 3, writeBank = 4;
constexpr unsigned programBank = VIRTUAL_BANK_NUM;
constexpr size_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);

int64_t offset(const DynamicMemRefType<float> &value, int64_t index) {
  int64_t result = value.offset;
  for (int64_t axis = value.rank - 1; axis >= 0; --axis) {
    result += (index % value.sizes[axis]) * value.strides[axis];
    index /= value.sizes[axis];
  }
  return result;
}
} // namespace

static void unary(UnrankedMemRefType<float> *output,
                  UnrankedMemRefType<float> *input,
                  const images::KernelImage &image) {
  DynamicMemRefType<float> out(*output), in(*input);
  if (out.rank != in.rank)
    abort();
  int64_t count = 1;
  for (int64_t axis = in.rank - 1; axis >= 0; --axis) {
    if (in.sizes[axis] <= 0 || out.sizes[axis] != in.sizes[axis])
      abort();
    count *= in.sizes[axis];
  }
  alignas(16) float source[bankBytes / sizeof(float)];
  alignas(16) float result[bankBytes / sizeof(float)];
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(0, readBank);
  bb_mem_transfer(1, writeBank);
  bb_mem_transfer(2, writeBank);
  mvin_kernel(image.bytes, image.size, programBank);
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding)::"memory");
  for (int64_t begin = 0; begin < count; begin += bankBytes / sizeof(float)) {
    uint32_t length =
        std::min<int64_t>(bankBytes / sizeof(float), count - begin);
    for (uint32_t i = 0; i < length; ++i)
      std::memcpy(source + i, in.data + offset(in, begin + i), sizeof(float));
    std::fill(source + length, source + (length + 3) / 4 * 4, 0.0f);
    bb_mvin_group((uintptr_t)source, readBank, 0, (length + 3) / 4, 1);
    alignas(16) kernel_launch packet{
        image.entry,
        image.text_bytes,
        0x40002000,
        {uint64_t(2) << 32, 0, length,
         (uint64_t(1) << 32) + offsetof(kernel_launch, reserved),
         rounding << 5},
        0};
    bb_mvin_group((uintptr_t)&packet, writeBank, 0, sizeof(packet) / 16, 1);
    run_kernel(readBank, programBank, writeBank, 0);
    bb_mvout_group((uintptr_t)result, writeBank, 1, (length + 3) / 4, 1);
    bb_mvout_group((uintptr_t)&packet, writeBank, 0, sizeof(packet) / 16, 1);
    asm volatile("csrs fflags, %0" ::"r"(packet.reserved) : "memory");
    for (uint32_t i = 0; i < length; ++i)
      std::memcpy(out.data + offset(out, begin + i), result + i, sizeof(float));
  }
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank);
}

extern "C" void _mlir_ciface_rvv_tanh(UnrankedMemRefType<float> *output,
                                      UnrankedMemRefType<float> *input) {
  unary(output, input, images::tanh);
}
extern "C" void _mlir_ciface_rvv_sin(UnrankedMemRefType<float> *output,
                                     UnrankedMemRefType<float> *input) {
  unary(output, input, images::sin);
}
extern "C" void _mlir_ciface_rvv_cos(UnrankedMemRefType<float> *output,
                                     UnrankedMemRefType<float> *input) {
  unary(output, input, images::cos);
}
