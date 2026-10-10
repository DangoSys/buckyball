#include "images.h"
#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <params.h>

namespace {
constexpr unsigned readBank = 5, writeBank = 6;
constexpr unsigned inputBank = 0, weightBank = 1, biasBank = 2;
constexpr unsigned descriptorBank = 3, outputBank = 4;
constexpr uint64_t descriptorAddress = uint64_t(descriptorBank) << 32;
constexpr unsigned programBank = VIRTUAL_BANK_NUM;
constexpr uint32_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);

int64_t elements(const DynamicMemRefType<float> &value) {
  int64_t count = 1;
  for (int64_t axis = value.rank - 1; axis >= 0; --axis) {
    if (value.sizes[axis] <= 0 ||
        (value.sizes[axis] != 1 && value.strides[axis] != count)) {
      fputs("RVV normalization requires contiguous FP32 tensors\n", stderr);
      abort();
    }
    count *= value.sizes[axis];
  }
  return count;
}
void load(const images::KernelImage &image) {
  for (unsigned bank = 0; bank < 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  for (unsigned bank = 0; bank < 3; ++bank)
    bb_mem_transfer(bank, readBank);
  bb_mem_transfer(descriptorBank, writeBank);
  bb_mem_transfer(outputBank, writeBank);
  mvin_kernel(image.bytes, image.size, programBank);
}
void send(const float *input, unsigned bank, uint32_t count, float *staging) {
  if (count % 4) {
    std::memcpy(staging, input, count * sizeof(float));
    std::fill_n(staging + count, 4 - count % 4, 0.0f);
    input = staging;
  }
  bb_mvin_group((uintptr_t)input, readBank, bank, (count + 3) / 4, 1);
}
void launch(kernel_launch &descriptor) {
  bb_mvin_group((uintptr_t)&descriptor, writeBank, 0, sizeof(descriptor) / 16,
                1);
  run_kernel(readBank, programBank, writeBank, 0);
}
void receive(float *output, uint32_t count, float *staging) {
  bb_mvout_group((uintptr_t)(count % 4 ? staging : output), writeBank, 1,
                 (count + 3) / 4, 1);
  alignas(16) kernel_launch descriptor;
  bb_mvout_group((uintptr_t)&descriptor, writeBank, 0, sizeof(descriptor) / 16,
                 1);
  if (count % 4)
    std::memcpy(output, staging, count * sizeof(float));
  asm volatile("csrs fflags, %0" ::"r"(descriptor.reserved) : "memory");
}
void release() {
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank);
}
uint32_t roundingMode() {
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding)::"memory");
  return rounding << 5;
}
} // namespace

static void layernorm(UnrankedMemRefType<float> *output,
                      UnrankedMemRefType<float> *input,
                      UnrankedMemRefType<float> *weight,
                      UnrankedMemRefType<float> *bias, uint32_t multiplier,
                      uint32_t epsilon) {
  DynamicMemRefType<float> out(*output), in(*input), w(*weight);
  const int64_t count = elements(in);
  const int64_t width = in.sizes[in.rank - 1];
  if (elements(out) != count || elements(w) != width ||
      width > bankBytes / sizeof(float)) {
    fputs("RVV LayerNorm requires one complete row and weights in banks\n",
          stderr);
    abort();
  }
  alignas(16) float staging[bankBytes / sizeof(float)];
  const uint32_t rounding = roundingMode();
  load(images::layernorm);
  send(w.data + w.offset, weightBank, width, staging);
  if (bias) {
    DynamicMemRefType<float> b(*bias);
    if (elements(b) != width)
      abort();
    send(b.data + b.offset, biasBank, width, staging);
  }
  for (int64_t begin = 0; begin < count; begin += width) {
    send(in.data + in.offset + begin, inputBank, width, staging);
    alignas(16) kernel_launch call{
        images::layernorm.entry,
        images::layernorm.text_bytes,
        0x40002000,
        {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
         uint64_t(weightBank) << 32, uint64_t(biasBank) << 32,
         uint64_t(width) | (uint64_t(bias != nullptr) << 32),
         (uint64_t(epsilon) << 32) | multiplier,
         descriptorAddress + offsetof(kernel_launch, reserved), rounding},
        0};
    launch(call);
    receive(out.data + out.offset + begin, width, staging);
  }
  release();
}

extern "C" void _mlir_ciface_rvv_layernorm(UnrankedMemRefType<float> *output,
                                           UnrankedMemRefType<float> *input,
                                           UnrankedMemRefType<float> *weight,
                                           UnrankedMemRefType<float> *bias,
                                           uint32_t multiplier,
                                           uint32_t epsilon) {
  layernorm(output, input, weight, bias, multiplier, epsilon);
}

extern "C" void _mlir_ciface_rvv_layernorm_no_bias(
    UnrankedMemRefType<float> *output, UnrankedMemRefType<float> *input,
    UnrankedMemRefType<float> *weight, uint32_t multiplier, uint32_t epsilon) {
  layernorm(output, input, weight, nullptr, multiplier, epsilon);
}

extern "C" void _mlir_ciface_rvv_gelu(UnrankedMemRefType<float> *output,
                                      UnrankedMemRefType<float> *input) {
  DynamicMemRefType<float> out(*output), in(*input);
  const int64_t count = elements(out);
  int64_t inputCount = 1;
  bool contiguous = true;
  for (int64_t axis = in.rank - 1; axis >= 0; --axis) {
    if (in.sizes[axis] != 1 && in.strides[axis] != inputCount)
      contiguous = false;
    inputCount *= in.sizes[axis];
  }
  if (inputCount != count)
    abort();
  alignas(16) float staging[bankBytes / sizeof(float)];
  const uint32_t rounding = roundingMode();
  load(images::gelu);
  for (int64_t begin = 0; begin < count;) {
    const uint32_t length =
        std::min<int64_t>(bankBytes / sizeof(float), count - begin);
    if (contiguous) {
      send(in.data + in.offset + begin, inputBank, length, staging);
    } else {
      for (uint32_t index = 0; index < length; ++index) {
        int64_t remaining = begin + index, offset = in.offset;
        for (int64_t axis = in.rank - 1; axis >= 0; --axis) {
          offset += (remaining % in.sizes[axis]) * in.strides[axis];
          remaining /= in.sizes[axis];
        }
        staging[index] = in.data[offset];
      }
      std::fill_n(staging + length, (4 - length % 4) % 4, 0.0f);
      bb_mvin_group((uintptr_t)staging, readBank, inputBank, (length + 3) / 4,
                    1);
    }
    alignas(16) kernel_launch call{
        images::gelu.entry,
        images::gelu.text_bytes,
        0x40002000,
        {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32, length,
         descriptorAddress + offsetof(kernel_launch, reserved), rounding, 0, 0,
         0},
        0};
    launch(call);
    receive(out.data + out.offset + begin, length, staging);
    begin += length;
  }
  release();
}
