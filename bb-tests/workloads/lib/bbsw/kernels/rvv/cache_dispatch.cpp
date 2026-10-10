#include "images.h"
#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <params.h>

namespace {
constexpr unsigned readBank = 4, writeBank = 5;
constexpr unsigned programBank = VIRTUAL_BANK_NUM;
constexpr uint32_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);
constexpr int64_t chunk = bankBytes / 128 * 32;
static_assert(chunk > 0 && bankBytes <= 65536);

template <typename T> int64_t elements(const DynamicMemRefType<T> &value) {
  if (!value.rank || value.sizes[value.rank - 1] <= 0)
    abort();
  int64_t count = 1;
  for (int64_t axis = 0; axis < value.rank; ++axis) {
    if (value.sizes[axis] <= 0)
      abort();
    count *= value.sizes[axis];
  }
  return count;
}

template <typename T>
void copyRows(DynamicMemRefType<T> &value, T *packed, int64_t first,
              int64_t count, bool write) {
  const int64_t width = value.sizes[value.rank - 1];
  const int64_t stride = value.strides[value.rank - 1];
  for (int64_t done = 0; done < count;) {
    int64_t index = first + done;
    const int64_t length = std::min(count - done, width - index % width);
    int64_t offset = value.offset;
    for (int64_t axis = value.rank - 1; axis >= 0; --axis) {
      offset += index % value.sizes[axis] * value.strides[axis];
      index /= value.sizes[axis];
    }
    T *row = value.data + offset;
    if (stride == 1) {
      if (write)
        std::memcpy(row, packed + done, length * sizeof(T));
      else
        std::memcpy(packed + done, row, length * sizeof(T));
    } else {
      for (int64_t column = 0; column < length; ++column) {
        if (write)
          row[column * stride] = packed[done + column];
        else
          packed[done + column] = row[column * stride];
      }
    }
    done += length;
  }
}

void check(DynamicMemRefType<int8_t> &codes,
           DynamicMemRefType<int8_t> &scales) {
  if (codes.rank != scales.rank || codes.sizes[codes.rank - 1] % 32)
    abort();
  for (int64_t axis = 0; axis < codes.rank; ++axis)
    if (scales.sizes[axis] !=
        codes.sizes[axis] / (axis == codes.rank - 1 ? 32 : 1))
      abort();
}

void load(const images::KernelImage &image, bool encoding) {
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  if (encoding) {
    bb_mem_transfer(3, readBank);
    bb_mem_transfer(0, writeBank);
    bb_mem_transfer(1, writeBank);
    bb_mem_transfer(2, writeBank);
  } else {
    bb_mem_transfer(2, readBank);
    bb_mem_transfer(3, readBank);
    bb_mem_transfer(0, writeBank);
    bb_mem_transfer(1, writeBank);
  }
  mvin_kernel(image.bytes, image.size, programBank);
}

void launch(const images::KernelImage &image, uint32_t count, bool encoding) {
  const uint64_t descriptor = uint64_t(encoding ? 1 : 2) << 32;
  alignas(16) kernel_launch packet{};
  packet.entry = image.entry;
  packet.end = image.text_bytes;
  packet.stack = 0x40002000;
  packet.args[0] = uint64_t(encoding ? 2 : 3) << 32;
  packet.args[1] = encoding ? uint64_t(3) << 32 : 0;
  packet.args[2] = encoding ? 0 : uint64_t(1) << 32;
  packet.args[3] = count;
  packet.args[4] = descriptor + offsetof(kernel_launch, reserved);
  bb_mvin_group((uintptr_t)&packet, writeBank, 0, sizeof(packet) / 16, 1);
  run_kernel(readBank, programBank, writeBank, 0);
  alignas(16) kernel_launch completed;
  bb_mvout_group((uintptr_t)&completed, writeBank, 0, sizeof(completed) / 16,
                 1);
  if (std::memcmp(&packet, &completed, offsetof(kernel_launch, reserved)) ||
      completed.reserved) {
    fputs("MXFP8 cache kernel rejected non-finite values, NaN encoding, or "
          "FP32 overflow\n",
          stderr);
    abort();
  }
}

void release() {
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank);
}
} // namespace

extern "C" void
_mlir_ciface_rvv_mxfp8_encode(UnrankedMemRefType<int8_t> *codes,
                              UnrankedMemRefType<int8_t> *scales,
                              UnrankedMemRefType<float> *input) {
  DynamicMemRefType<int8_t> c(*codes), s(*scales);
  DynamicMemRefType<float> in(*input);
  check(c, s);
  int64_t count = elements(in);
  if (elements(c) != count || elements(s) != count / 32 || in.rank != c.rank)
    abort();
  for (int64_t axis = 0; axis < in.rank; ++axis)
    if (in.sizes[axis] != c.sizes[axis])
      abort();
  alignas(16) float staging[chunk];
  alignas(16) int8_t codeData[chunk];
  alignas(16) int8_t scaleData[(chunk / 32 + 15) / 16 * 16];
  load(images::cache_quant, true);
  for (int64_t begin = 0; begin < count; begin += chunk) {
    uint32_t length = std::min(chunk, count - begin);
    copyRows(in, staging, begin, length, false);
    bb_mvin_group((uintptr_t)staging, readBank, 0, length / 4, 1);
    launch(images::cache_quant, length, true);
    bb_mvout_group((uintptr_t)codeData, writeBank, 1, length / 16, 1);
    bb_mvout_group((uintptr_t)scaleData, writeBank, 2, (length / 32 + 15) / 16,
                   1);
    copyRows(c, codeData, begin, length, true);
    copyRows(s, scaleData, begin / 32, length / 32, true);
  }
  release();
}

extern "C" void
_mlir_ciface_rvv_mxfp8_decode(UnrankedMemRefType<float> *output,
                              UnrankedMemRefType<int8_t> *codes,
                              UnrankedMemRefType<int8_t> *scales) {
  DynamicMemRefType<float> out(*output);
  DynamicMemRefType<int8_t> c(*codes), s(*scales);
  check(c, s);
  int64_t count = elements(c);
  if (elements(out) != count || elements(s) != count / 32 || out.rank != c.rank)
    abort();
  for (int64_t axis = 0; axis < out.rank; ++axis)
    if (out.sizes[axis] != c.sizes[axis])
      abort();
  alignas(16) float staging[chunk];
  alignas(16) int8_t codeData[chunk];
  alignas(16) int8_t scaleData[(chunk / 32 + 15) / 16 * 16]{};
  load(images::dequant, false);
  for (int64_t begin = 0; begin < count; begin += chunk) {
    uint32_t length = std::min(chunk, count - begin);
    copyRows(c, codeData, begin, length, false);
    copyRows(s, scaleData, begin / 32, length / 32, false);
    bb_mvin_group((uintptr_t)codeData, readBank, 0, length / 16, 1);
    bb_mvin_group((uintptr_t)scaleData, readBank, 1, (length / 32 + 15) / 16,
                  1);
    launch(images::dequant, length, false);
    bb_mvout_group((uintptr_t)staging, writeBank, 1, length / 4, 1);
    copyRows(out, staging, begin, length, true);
  }
  release();
}
