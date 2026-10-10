#include "images.h"
#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdlib>
#include <cstring>
#include <params.h>

namespace {
constexpr unsigned readBank = 4, writeBank = 5;
constexpr unsigned programBank = VIRTUAL_BANK_NUM;
constexpr size_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);
constexpr size_t chunk = bankBytes / sizeof(float);

bool linear(const DynamicMemRefType<float> &value,
            const DynamicMemRefType<float> &shape) {
  if (value.rank != shape.rank)
    return false;
  int64_t stride = 1;
  for (int64_t axis = value.rank - 1; axis >= 0; --axis) {
    if (value.sizes[axis] != shape.sizes[axis] ||
        (value.sizes[axis] != 1 && value.strides[axis] != stride))
      return false;
    stride *= value.sizes[axis];
  }
  return true;
}

int64_t offset(const DynamicMemRefType<float> &value,
               const DynamicMemRefType<float> &shape, int64_t index) {
  int64_t result = value.offset;
  for (int64_t axis = shape.rank - 1; axis >= 0; --axis) {
    int64_t coordinate = index % shape.sizes[axis];
    index /= shape.sizes[axis];
    int64_t inputAxis = axis - (shape.rank - value.rank);
    if (inputAxis >= 0 && value.sizes[inputAxis] != 1)
      result += coordinate * value.strides[inputAxis];
  }
  return result;
}

void run(UnrankedMemRefType<float> *output, UnrankedMemRefType<float> *lhs,
         UnrankedMemRefType<float> *rhs, uint32_t operation) {
  DynamicMemRefType<float> out(*output), a(*lhs);
  int64_t count = 1;
  for (int64_t axis = 0; axis < out.rank; ++axis) {
    if (out.sizes[axis] <= 0)
      abort();
    count *= out.sizes[axis];
  }
  for (auto *input : {lhs, rhs}) {
    if (!input)
      continue;
    DynamicMemRefType<float> value(*input);
    if (value.rank > out.rank)
      abort();
    for (int64_t axis = 0; axis < value.rank; ++axis)
      if (value.sizes[axis] != 1 &&
          value.sizes[axis] != out.sizes[axis + out.rank - value.rank])
        abort();
  }
  alignas(16) float aTile[chunk], bTile[chunk], result[chunk];
  const bool outputLinear = linear(out, out);
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding)::"memory");
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(2, readBank);
  bb_mem_transfer(3, readBank);
  bb_mem_transfer(0, writeBank);
  bb_mem_transfer(1, writeBank);
  mvin_kernel(images::pointwise.bytes, images::pointwise.size, programBank);
  for (int64_t first = 0; first < count; first += chunk) {
    uint32_t length = std::min<int64_t>(chunk, count - first);
    auto pack = [&](const DynamicMemRefType<float> &value, float *staging) {
      if (linear(value, out)) {
        const float *source = value.data + value.offset + first;
        if (length % 4 == 0)
          return source;
        std::memcpy(staging, source, length * sizeof(float));
      } else {
        bool scalar = true;
        for (int64_t axis = 0; axis < value.rank; ++axis)
          scalar &= value.sizes[axis] == 1;
        if (scalar)
          std::fill_n(staging, length, value.data[value.offset]);
        else
          for (uint32_t index = 0; index < length; ++index)
            staging[index] = value.data[offset(value, out, first + index)];
      }
      std::fill(staging + length, staging + (length + 3) / 4 * 4, 0.0f);
      return static_cast<const float *>(staging);
    };
    bb_mvin_group((uintptr_t)pack(a, aTile), readBank, 0, (length + 3) / 4, 1);
    if (rhs) {
      DynamicMemRefType<float> b(*rhs);
      bb_mvin_group((uintptr_t)pack(b, bTile), readBank, 1, (length + 3) / 4,
                    1);
    }
    alignas(16) kernel_launch call{
        images::pointwise.entry,
        images::pointwise.text_bytes,
        0x40002000,
        {uint64_t(3) << 32, 0, uint64_t(1) << 32, length, operation,
         (uint64_t(2) << 32) + offsetof(kernel_launch, reserved),
         rounding << 5},
        0};
    bb_mvin_group((uintptr_t)&call, writeBank, 0, sizeof(call) / 16, 1);
    run_kernel(readBank, programBank, writeBank, 0);
    float *destination = outputLinear && length % 4 == 0
                             ? out.data + out.offset + first
                             : result;
    bb_mvout_group((uintptr_t)destination, writeBank, 1, (length + 3) / 4, 1);
    bb_mvout_group((uintptr_t)&call, writeBank, 0, sizeof(call) / 16, 1);
    asm volatile("csrs fflags, %0" ::"r"(call.reserved) : "memory");
    if (destination == result) {
      if (outputLinear)
        std::memcpy(out.data + out.offset + first, result,
                    length * sizeof(float));
      else
        for (uint32_t index = 0; index < length; ++index)
          out.data[offset(out, out, first + index)] = result[index];
    }
  }
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank);
}
} // namespace

extern "C" void _mlir_ciface_rvv_binary(UnrankedMemRefType<float> *output,
                                        UnrankedMemRefType<float> *lhs,
                                        UnrankedMemRefType<float> *rhs,
                                        uint32_t operation) {
  if (operation > 7 || operation == 4 || operation == 5)
    abort();
  run(output, lhs, rhs, operation);
}
extern "C" void _mlir_ciface_rvv_unary(UnrankedMemRefType<float> *output,
                                       UnrankedMemRefType<float> *input,
                                       uint32_t operation) {
  if (operation < 4 || operation > 5)
    abort();
  run(output, input, nullptr, operation);
}
