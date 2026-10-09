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
constexpr unsigned outputBank = 3, inputBank = 0, upBank = 1;
constexpr uint64_t descriptorAddress = uint64_t(2) << 32;
constexpr unsigned programBank = VIRTUAL_BANK_NUM;
constexpr int64_t chunk = 1024;
constexpr uint32_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);
static_assert(bankBytes >= 4096 && bankBytes <= 65536 && bankBytes % 16 == 0);

int64_t elements(const DynamicMemRefType<float> &value) {
  int64_t count = 1;
  for (int64_t axis = value.rank - 1; axis >= 0; --axis) {
    if (value.sizes[axis] <= 0 ||
        (value.sizes[axis] != 1 && value.strides[axis] != count)) {
      fputs("RVV kernel requires a positive contiguous FP32 tensor\n", stderr);
      abort();
    }
    count *= value.sizes[axis];
  }
  return count;
}

void load(const images::KernelImage &image, unsigned buffer) {
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(2, readBank);
  bb_mem_transfer(3, readBank);
  bb_mem_transfer(0, writeBank);
  bb_mem_transfer(1, writeBank);
  mvin_kernel(image.bytes, image.size, programBank + buffer);
}

uint32_t roundingMode() {
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding)::"memory");
  return rounding << 5;
}

void launch(const void *descriptor, uint32_t bytes, unsigned buffer) {
  bb_mvin_group((uintptr_t)descriptor, writeBank, 0, bytes / 16, 1);
  run_kernel(readBank, programBank + buffer, writeBank, 0);
}

void send(const float *input, unsigned bank, uint32_t count, float *staging) {
  if (count % 4) {
    std::memcpy(staging, input, count * sizeof(float));
    std::fill_n(staging + count, 4 - count % 4, 0.0f);
    input = staging;
  }
  bb_mvin_group((uintptr_t)input, readBank, bank, (count + 3) / 4, 1);
}

void receive(float *output, uint32_t count, float *staging) {
  bb_mvout_group((uintptr_t)(count % 4 ? staging : output), writeBank, 1,
                 (count + 3) / 4, 1);
  if (count % 4)
    std::memcpy(output, staging, count * sizeof(float));
}

// Tensor launch entries return private exception flags in the launch descriptor
// reserved field.
void receiveTensor(float *output, uint32_t count, float *staging) {
  receive(output, count, staging);
  alignas(16) kernel_launch descriptor;
  bb_mvout_group((uintptr_t)&descriptor, writeBank, 0, sizeof(descriptor) / 16,
                 1);
  asm volatile("csrs fflags, %0" ::"r"(descriptor.reserved) : "memory");
}

void release(unsigned buffer) {
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank + buffer);
}
} // namespace

extern "C" void _mlir_ciface_rvv_silu(UnrankedMemRefType<float> *output,
                                      UnrankedMemRefType<float> *input) {
  DynamicMemRefType<float> out(*output), in(*input);
  int64_t count = elements(in);
  if (elements(out) != count)
    abort();
  alignas(16) float staging[chunk];
  const uint32_t rounding = roundingMode();
  load(images::silu, 0);
  for (int64_t begin = 0; begin < count; begin += chunk) {
    uint32_t length = std::min(chunk, count - begin);
    send(in.data + in.offset + begin, inputBank, length, staging);
    kernel_launch call{
        images::silu.entry,
        images::silu.text_bytes,
        0x40002000,
        {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32, length,
         descriptorAddress + offsetof(kernel_launch, reserved), rounding},
        0};
    launch(&call, sizeof(call), 0);
    receiveTensor(out.data + out.offset + begin, length, staging);
  }
  release(0);
}

extern "C" void _mlir_ciface_rvv_swiglu(UnrankedMemRefType<float> *output,
                                        UnrankedMemRefType<float> *gate,
                                        UnrankedMemRefType<float> *up) {
  DynamicMemRefType<float> out(*output), a(*gate), b(*up);
  int64_t count = elements(a);
  if (elements(out) != count || elements(b) != count)
    abort();
  alignas(16) float staging[chunk];
  const uint32_t rounding = roundingMode();
  load(images::swiglu, 0);
  for (int64_t begin = 0; begin < count; begin += chunk) {
    uint32_t length = std::min(chunk, count - begin);
    send(a.data + a.offset + begin, inputBank, length, staging);
    send(b.data + b.offset + begin, upBank, length, staging);
    kernel_launch call{images::swiglu.entry,
                       images::swiglu.text_bytes,
                       0x40002000,
                       {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
                        uint64_t(upBank) << 32, length,
                        descriptorAddress + offsetof(kernel_launch, reserved),
                        rounding},
                       0};
    launch(&call, sizeof(call), 0);
    receiveTensor(out.data + out.offset + begin, length, staging);
  }
  release(0);
}

extern "C" void _mlir_ciface_rvv_snake(UnrankedMemRefType<float> *output,
                                       UnrankedMemRefType<float> *input,
                                       UnrankedMemRefType<float> *logAlpha,
                                       UnrankedMemRefType<float> *logBeta) {
  DynamicMemRefType<float> out(*output), in(*input), alpha(*logAlpha),
      beta(*logBeta);
  int64_t channels = elements(alpha), count = elements(in);
  if (elements(beta) != channels || elements(out) != count || count % channels)
    abort();
  int64_t width = count / channels;
  alignas(16) float staging[chunk];
  load(images::snake, 1);
  // Short channels fit together in one input bank and one coefficient packet.
  const int64_t batch =
      width <= chunk
          ? std::min<int64_t>(chunk / width,
                              (bankBytes - sizeof(kernel_launch)) / 8)
          : 1;
  alignas(16) uint8_t descriptor[bankBytes];
  for (int64_t channel = 0; channel < channels; channel += batch) {
    uint32_t grouped = std::min(batch, channels - channel);
    for (int64_t begin = 0; begin < width; begin += chunk) {
      uint32_t length = std::min(chunk, width - begin);
      int64_t offset = channel * width + begin;
      uint32_t elements = grouped * length;
      send(in.data + in.offset + offset, inputBank, elements, staging);
      kernel_launch call{
          images::snake.entry,
          images::snake.text_bytes,
          0x40002000,
          {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32, grouped,
           length, descriptorAddress + sizeof(kernel_launch),
           descriptorAddress + sizeof(kernel_launch) + grouped * 4},
          0};
      uint32_t bytes = (sizeof(call) + grouped * 2 * sizeof(float) + 15) & ~15U;
      std::memset(descriptor, 0, bytes);
      std::memcpy(descriptor, &call, sizeof(call));
      std::memcpy(descriptor + sizeof(call),
                  alpha.data + alpha.offset + channel, grouped * sizeof(float));
      std::memcpy(descriptor + sizeof(call) + grouped * sizeof(float),
                  beta.data + beta.offset + channel, grouped * sizeof(float));
      launch(descriptor, bytes, 1);
      receive(out.data + out.offset + offset, elements, staging);
    }
  }
  release(1);
}

extern "C" void _mlir_ciface_rvv_norm(UnrankedMemRefType<float> *output,
                                      UnrankedMemRefType<float> *input,
                                      UnrankedMemRefType<float> *weight,
                                      uint32_t meanMultiplierBits,
                                      uint32_t epsilonBits) {
  DynamicMemRefType<float> out(*output), in(*input), w(*weight);
  if (in.rank < 1 || in.strides[in.rank - 1] != 1)
    abort();
  int64_t count = 1;
  for (int64_t axis = 0; axis < in.rank; ++axis) {
    if (in.sizes[axis] <= 0)
      abort();
    count *= in.sizes[axis];
  }
  if (elements(out) != count)
    abort();
  int64_t width = in.sizes[in.rank - 1];
  if (elements(w) != width || width > bankBytes / sizeof(float)) {
    fputs("RVV norm requires one complete row and its weights in banks\n",
          stderr);
    abort();
  }
  alignas(16) float staging[bankBytes / sizeof(float)];
  const uint32_t rounding = roundingMode();
  load(images::norm, 0);
  send(w.data + w.offset, upBank, width, staging);
  for (int64_t begin = 0; begin < count; begin += width) {
    int64_t row = begin / width, offset = in.offset;
    for (int64_t axis = in.rank - 2; axis >= 0; --axis) {
      offset += (row % in.sizes[axis]) * in.strides[axis];
      row /= in.sizes[axis];
    }
    send(in.data + offset, inputBank, width, staging);
    kernel_launch call{images::norm.entry,
                       images::norm.text_bytes,
                       0x40002000,
                       {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
                        uint64_t(upBank) << 32, uint32_t(width),
                        meanMultiplierBits, epsilonBits,
                        descriptorAddress + offsetof(kernel_launch, reserved),
                        rounding},
                       0};
    launch(&call, sizeof(call), 0);
    receiveTensor(out.data + out.offset + begin, width, staging);
  }
  release(0);
}

extern "C" void _mlir_ciface_rvv_norm_no_weight(
    UnrankedMemRefType<float> *output, UnrankedMemRefType<float> *input,
    uint32_t meanMultiplierBits, uint32_t epsilonBits) {
  DynamicMemRefType<float> out(*output), in(*input);
  if (in.rank < 1 || in.strides[in.rank - 1] != 1)
    abort();
  int64_t count = 1;
  for (int64_t axis = 0; axis < in.rank; ++axis) {
    if (in.sizes[axis] <= 0)
      abort();
    count *= in.sizes[axis];
  }
  if (elements(out) != count)
    abort();
  int64_t width = in.sizes[in.rank - 1];
  if (width > bankBytes / sizeof(float)) {
    fputs("RVV norm requires one complete row in a bank\n", stderr);
    abort();
  }
  alignas(16) float staging[bankBytes / sizeof(float)];
  const uint32_t rounding = roundingMode();
  load(images::norm_no_weight, 0);
  for (int64_t begin = 0; begin < count; begin += width) {
    int64_t row = begin / width, offset = in.offset;
    for (int64_t axis = in.rank - 2; axis >= 0; --axis) {
      offset += (row % in.sizes[axis]) * in.strides[axis];
      row /= in.sizes[axis];
    }
    send(in.data + offset, inputBank, width, staging);
    kernel_launch call{images::norm_no_weight.entry,
                       images::norm_no_weight.text_bytes,
                       0x40002000,
                       {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
                        uint32_t(width), meanMultiplierBits, epsilonBits,
                        descriptorAddress + offsetof(kernel_launch, reserved),
                        rounding},
                       0};
    launch(&call, sizeof(call), 0);
    receiveTensor(out.data + out.offset + begin, width, staging);
  }
  release(0);
}

static void tensorSoftmax(UnrankedMemRefType<float> *output,
                          UnrankedMemRefType<float> *input,
                          const images::KernelImage &image) {
  DynamicMemRefType<float> out(*output), in(*input);
  int64_t count = elements(in);
  if (in.rank < 1 || elements(out) != count)
    abort();
  int64_t width = in.sizes[in.rank - 1];
  if (width > bankBytes / sizeof(float)) {
    fputs("RVV softmax requires one complete row in a bank\n", stderr);
    abort();
  }
  int64_t rows = count / width, batch = bankBytes / sizeof(float) / width;
  alignas(16) float staging[bankBytes / sizeof(float)];
  const uint32_t rounding = roundingMode();
  load(image, 0);
  for (int64_t row = 0; row < rows; row += batch) {
    uint32_t grouped = std::min(batch, rows - row), length = grouped * width;
    send(in.data + in.offset + row * width, inputBank, length, staging);
    kernel_launch call{image.entry,
                       image.text_bytes,
                       0x40002000,
                       {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
                        grouped, uint32_t(width),
                        descriptorAddress + offsetof(kernel_launch, reserved),
                        rounding},
                       0};
    launch(&call, sizeof(call), 0);
    receiveTensor(out.data + out.offset + row * width, length, staging);
  }
  release(0);
}

extern "C" void _mlir_ciface_rvv_softmax(UnrankedMemRefType<float> *output,
                                         UnrankedMemRefType<float> *input) {
  tensorSoftmax(output, input, images::softmax);
}

extern "C" void
_mlir_ciface_rvv_logsumexp_softmax(UnrankedMemRefType<float> *output,
                                   UnrankedMemRefType<float> *input) {
  tensorSoftmax(output, input, images::logsumexp_softmax);
}

extern "C" void
_mlir_ciface_rvv_attention_softmax(UnrankedMemRefType<float> *output,
                                   UnrankedMemRefType<float> *input,
                                   UnrankedMemRefType<uint8_t> *mask,
                                   uint32_t scaleBits, uint32_t maskedBits) {
  DynamicMemRefType<float> out(*output), in(*input);
  DynamicMemRefType<uint8_t> condition(*mask);
  int64_t count = elements(in);
  if (in.rank != 4 || in.sizes[0] != 1 || elements(out) != count ||
      condition.rank != 4 || condition.sizes[0] != 1 ||
      condition.sizes[1] != 1 ||
      (condition.sizes[2] != 1 && condition.sizes[2] != in.sizes[2]) ||
      condition.sizes[3] != in.sizes[3] || condition.strides[3] != 1 ||
      (condition.sizes[2] != 1 && condition.strides[2] != in.sizes[3]))
    abort();
  int64_t width = in.sizes[3];
  int64_t maskQueries = condition.sizes[2];
  if (width > bankBytes / sizeof(float) || maskQueries * width > bankBytes) {
    fputs("RVV attention softmax requires one complete row in a bank\n",
          stderr);
    abort();
  }
  int64_t rows = count / width, batch = bankBytes / sizeof(float) / width;
  alignas(16) float staging[bankBytes / sizeof(float)];
  alignas(16) uint8_t maskStaging[bankBytes];
  const uint32_t rounding = roundingMode();
  load(images::attention_softmax, 0);
  uint32_t maskBytes = maskQueries * width;
  std::memcpy(maskStaging, condition.data + condition.offset, maskBytes);
  std::memset(maskStaging + maskBytes, 0, (16 - maskBytes % 16) % 16);
  bb_mvin_group((uintptr_t)maskStaging, readBank, upBank, (maskBytes + 15) / 16,
                1);
  for (int64_t row = 0; row < rows;) {
    uint32_t grouped = std::min(batch, rows - row);
    uint32_t length = grouped * width;
    send(in.data + in.offset + row * width, inputBank, length, staging);
    alignas(16)
        uint8_t descriptor[sizeof(kernel_launch) + 4 * sizeof(uint32_t)]{};
    kernel_launch call{images::attention_softmax.entry,
                       images::attention_softmax.text_bytes,
                       0x40002000,
                       {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
                        uint64_t(upBank) << 32, grouped, uint32_t(width),
                        descriptorAddress + sizeof(kernel_launch),
                        descriptorAddress + offsetof(kernel_launch, reserved),
                        rounding},
                       0};
    std::memcpy(descriptor, &call, sizeof(call));
    uint32_t parameters[]{scaleBits, maskedBits, uint32_t(maskQueries),
                          uint32_t(row % maskQueries)};
    std::memcpy(descriptor + sizeof(call), parameters, sizeof(parameters));
    launch(descriptor, sizeof(descriptor), 0);
    receiveTensor(out.data + out.offset + row * width, length, staging);
    row += grouped;
  }
  release(0);
}

extern "C" void _mlir_ciface_rvv_rope(UnrankedMemRefType<float> *output,
                                      UnrankedMemRefType<float> *input,
                                      UnrankedMemRefType<float> *frequencies,
                                      UnrankedMemRefType<int64_t> *positions,
                                      uint32_t scaleBits) {
  DynamicMemRefType<float> out(*output), in(*input), freq(*frequencies);
  DynamicMemRefType<int64_t> pos(*positions);
  int64_t count = elements(in);
  if (in.rank != 4 || in.sizes[0] != 1 || elements(out) != count)
    abort();
  int64_t heads = in.sizes[1], rows = in.sizes[2], width = in.sizes[3];
  if (width % 2 || elements(freq) != width / 2 || pos.rank != 1 ||
      pos.sizes[0] != rows || pos.strides[0] != 1 ||
      width > bankBytes / sizeof(float)) {
    fputs(
        "RVV RoPE requires complete even-width heads and matching positions\n",
        stderr);
    abort();
  }
  int64_t batch =
      std::min<int64_t>(bankBytes / sizeof(float) / width,
                        (bankBytes - sizeof(kernel_launch)) / sizeof(int32_t));
  alignas(16) float staging[bankBytes / sizeof(float)];
  alignas(16) uint8_t descriptor[bankBytes];
  const uint32_t rounding = roundingMode();
  load(images::rope, 0);
  send(freq.data + freq.offset, upBank, width / 2, staging);
  const int64_t rowChunk = std::min(batch, rows);
  const int64_t headChunk = bankBytes / sizeof(float) / (rowChunk * width);
  for (int64_t head = 0; head < heads; head += headChunk)
    for (int64_t row = 0; row < rows; row += rowChunk) {
      uint32_t groupedRows = std::min(rowChunk, rows - row);
      uint32_t groupedHeads = std::min(headChunk, heads - head);
      uint32_t length = groupedHeads * groupedRows * width;
      int64_t begin = (head * rows + row) * width;
      send(in.data + in.offset + begin, inputBank, length, staging);
      kernel_launch call{
          images::rope.entry,
          images::rope.text_bytes,
          0x40002000,
          {uint64_t(outputBank) << 32, uint64_t(inputBank) << 32,
           uint64_t(upBank) << 32, descriptorAddress + sizeof(kernel_launch),
           groupedRows,
           (uint64_t(scaleBits) << 32) | (groupedHeads << 16) | uint32_t(width),
           descriptorAddress + offsetof(kernel_launch, reserved), rounding},
          0};
      uint32_t bytes =
          (sizeof(call) + groupedRows * sizeof(int32_t) + 15) & ~15U;
      std::memset(descriptor, 0, bytes);
      std::memcpy(descriptor, &call, sizeof(call));
      auto packedPositions =
          reinterpret_cast<int32_t *>(descriptor + sizeof(call));
      for (size_t index = 0; index < groupedRows; ++index) {
        const int64_t position = pos.data[pos.offset + row + index];
        if (position < INT32_MIN || position > INT32_MAX)
          abort();
        packedPositions[index] = static_cast<int32_t>(position);
      }
      launch(descriptor, bytes, 0);
      receiveTensor(out.data + out.offset + begin, length, staging);
    }
  release(0);
}
