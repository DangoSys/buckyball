#include "flash_attention.h"
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
constexpr unsigned programBank = VIRTUAL_BANK_NUM;
constexpr size_t bankBytes = BANK_LINES * (BANK_WIDTH / 8);
struct alignas(16) Packet {
  kernel_launch launch;
  uint64_t padding[4];
  float mask[rvv_flash::queries * rvv_flash::lanes];
};
static_assert(offsetof(Packet, mask) == rvv_flash::maskOffset);
static_assert(sizeof(Packet) == rvv_flash::stateOffset);
static_assert(rvv_flash::stateOffset + sizeof(rvv_flash::State) <= bankBytes);

void panel(const DynamicMemRefType<float> &source, int64_t batch, int64_t head,
           int64_t begin, unsigned rows, unsigned width, float *staging) {
  for (unsigned row = 0; row < rows; ++row)
    std::memcpy(staging + row * width,
                source.data + source.offset + batch * source.strides[0] +
                    head * source.strides[1] +
                    (begin + row) * source.strides[2],
                width * sizeof(float));
}
} // namespace

extern "C" void _mlir_ciface_rvv_flash_attention(
    UnrankedMemRefType<float> *output, UnrankedMemRefType<float> *scores,
    UnrankedMemRefType<float> *query, UnrankedMemRefType<float> *key,
    UnrankedMemRefType<float> *value, UnrankedMemRefType<float> *mask,
    uint32_t scaleBits) {
  DynamicMemRefType<float> out(*output), score(*scores), q(*query), k(*key),
      v(*value), condition(*mask);
  if (q.rank != 4 || k.rank != 4 || v.rank != 4 || out.rank != 4 ||
      score.rank != 3 || condition.rank != 4)
    abort();
  const int64_t batches = q.sizes[0], heads = q.sizes[1], queries = q.sizes[2];
  const int64_t keys = k.sizes[2], width = q.sizes[3];
  if (batches <= 0 || heads <= 0 || queries <= 0 || keys <= 0 || width <= 0 ||
      width % rvv_flash::lanes || width * sizeof(float) > bankBytes ||
      k.sizes[0] != batches || v.sizes[0] != batches || k.sizes[1] != heads ||
      v.sizes[1] != heads || v.sizes[2] != keys || k.sizes[3] != width ||
      v.sizes[3] != width || condition.sizes[0] != batches ||
      condition.sizes[1] != 1 || condition.sizes[2] != queries ||
      condition.sizes[3] != keys || q.strides[3] != 1 || k.strides[3] != 1 ||
      v.strides[3] != 1 || condition.strides[3] != 1 || out.strides[3] != 1 ||
      score.strides[2] != 1)
    abort();
  for (unsigned axis = 0; axis < 4; ++axis)
    if (out.sizes[axis] != q.sizes[axis] ||
        (axis < 3 && score.sizes[axis] != q.sizes[axis]))
      abort();
  const unsigned tileRows =
      std::min<size_t>(rvv_flash::queries, bankBytes / (width * sizeof(float)));
  const unsigned tileKeys =
      std::min<size_t>(rvv_flash::lanes, bankBytes / (width * sizeof(float)));
  alignas(16) float staging[bankBytes / sizeof(float)];
  Packet packet{};
  packet.launch.entry = images::flash_attention.entry;
  packet.launch.end = images::flash_attention.text_bytes;
  packet.launch.stack = 0x40002000;
  packet.launch.args[0] = 0;
  packet.launch.args[1] = uint64_t(1) << 32;
  packet.launch.args[2] = uint64_t(2) << 32;
  packet.launch.args[3] = uint64_t(3) << 32;
  packet.launch.args[4] = uint64_t(4) << 32;
  packet.launch.args[7] =
      (uint64_t(2) << 32) + offsetof(kernel_launch, reserved);
  uint32_t rounding;
  asm volatile("csrr %0, frm" : "=r"(rounding)::"memory");
  packet.launch.args[6] = uint64_t(scaleBits) | (uint64_t(rounding << 5) << 32);
  for (unsigned bank = 0; bank < 5; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(0, readBank);
  bb_mem_transfer(1, readBank);
  for (unsigned bank = 2; bank < 5; ++bank)
    bb_mem_transfer(bank, writeBank);
  mvin_kernel(images::flash_attention.bytes, images::flash_attention.size,
              programBank);
  auto launch = [&](unsigned phase, unsigned rows, unsigned keyCount,
                    unsigned begin = 0, unsigned count = 0) {
    packet.launch.args[5] = uint64_t(phase) | (uint64_t(rows) << 8) |
                            (uint64_t(width) << 16) | (uint64_t(begin) << 32) |
                            (uint64_t(count) << 40) |
                            (uint64_t(keyCount) << 48);
    packet.launch.reserved = 0;
    bb_mvin_group((uintptr_t)&packet, writeBank, 0, sizeof(packet) / 16, 1);
    run_kernel(readBank, programBank, writeBank, 0);
    bb_mvout_group(
        (uintptr_t)&packet, writeBank, 0,
        phase == 5 ? sizeof(packet) / 16 : sizeof(kernel_launch) / 16, 1);
    asm volatile("csrs fflags, %0" ::"r"(packet.launch.reserved) : "memory");
  };
  for (int64_t batch = 0; batch < batches; ++batch)
    for (int64_t head = 0; head < heads; ++head)
      for (int64_t queryBegin = 0; queryBegin < queries;
           queryBegin += tileRows) {
        unsigned rows = std::min<int64_t>(tileRows, queries - queryBegin);
        panel(q, batch, head, queryBegin, rows, width, staging);
        bb_mvin_group((uintptr_t)staging, readBank, 0, rows * width / 4, 1);
        launch(0, rows, 0);
        for (int64_t keyBegin = 0; keyBegin < keys;
             keyBegin += rvv_flash::keys) {
          unsigned keyCount =
              std::min<int64_t>(rvv_flash::keys, keys - keyBegin);
          for (unsigned begin = 0; begin < keyCount; begin += tileKeys) {
            unsigned count = std::min(tileKeys, keyCount - begin);
            panel(k, batch, head, keyBegin + begin, count, width, staging);
            bb_mvin_group((uintptr_t)staging, readBank, 1, count * width / 4,
                          1);
            for (unsigned row = 0; row < rows; ++row)
              std::memcpy(packet.mask + row * rvv_flash::lanes,
                          condition.data + condition.offset +
                              batch * condition.strides[0] +
                              (queryBegin + row) * condition.strides[2] +
                              keyBegin + begin,
                          count * sizeof(float));
            launch(1, rows, keyCount, begin, count);
          }
          launch(2, rows, keyCount);
          for (unsigned begin = 0; begin < keyCount; begin += tileKeys) {
            unsigned count = std::min(tileKeys, keyCount - begin);
            panel(v, batch, head, keyBegin + begin, count, width, staging);
            bb_mvin_group((uintptr_t)staging, readBank, 1, count * width / 4,
                          1);
            launch(3, rows, keyCount, begin, count);
          }
          launch(4, rows, keyCount);
        }
        launch(5, rows, 0);
        bb_mvout_group((uintptr_t)staging, writeBank, 1, rows * width / 4, 1);
        for (unsigned row = 0; row < rows; ++row) {
          std::memcpy(out.data + out.offset + batch * out.strides[0] +
                          head * out.strides[1] +
                          (queryBegin + row) * out.strides[2],
                      staging + row * width, width * sizeof(float));
          std::memcpy(score.data + score.offset + batch * score.strides[0] +
                          head * score.strides[1] + queryBegin + row,
                      packet.mask + row, sizeof(float));
        }
      }
  bb_mem_release(readBank);
  bb_mem_release(writeBank);
  release_kernel(programBank);
}
