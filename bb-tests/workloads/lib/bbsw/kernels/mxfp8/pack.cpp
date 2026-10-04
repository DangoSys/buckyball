#include <CRunnerUtils.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "quant.h"

extern "C" void _mlir_ciface_mxfp8_quant(StridedMemRefType<float, 2> *input,
                                         StridedMemRefType<int8_t, 1> *output,
                                         int64_t tileRows, int64_t tileK,
                                         int64_t panelStride) {
  int64_t m = input->sizes[0], k = input->sizes[1];
  int64_t mt = (m + tileRows - 1) / tileRows, kt = (k + tileK - 1) / tileK;
  int64_t bytes = panelStride;
  if (k % 32 || tileK % 32 || output->sizes[0] != mt * kt * bytes) {
    fprintf(stderr, "mxfp8: invalid activation packing shape\n");
    abort();
  }
  auto *source = input->data + input->offset;
  auto *dest = output->data + output->offset;
  const int64_t rowStride = input->strides[0], columnStride = input->strides[1];
  const int64_t outputStride = output->strides[0];
  for (int64_t mr = 0; mr < mt; ++mr)
    for (int64_t kr = 0; kr < kt; ++kr)
      for (int64_t row = 0; row < tileRows; ++row) {
        int64_t width = std::min(tileK, k - kr * tileK);
        int64_t base = (mr * kt + kr) * bytes;
        if (mr * tileRows + row >= m && outputStride == 1) {
          std::memset(dest + base + row * width, 0, width);
          std::memset(dest + base + tileRows * width + row * (width / 32), 127,
                      width / 32);
          continue;
        }
        alignas(16) uint32_t staging[1024];
        alignas(16) uint8_t packed[1056];
        for (int64_t begin = 0; begin < width; begin += 1024) {
          int64_t count = std::min<int64_t>(1024, width - begin);
          int64_t r = mr * tileRows + row, c = kr * tileK + begin;
          int64_t valid =
              r < m ? std::max<int64_t>(0, std::min(count, k - c)) : 0;
          const uint32_t *values;
          if (valid == count && columnStride == 1) {
            values =
                reinterpret_cast<const uint32_t *>(source + r * rowStride + c);
          } else {
            for (int64_t i = 0; i < valid; ++i)
              staging[i] = __builtin_bit_cast(
                  uint32_t, source[r * rowStride + (c + i) * columnStride]);
            std::fill(staging + valid, staging + count, 0);
            values = staging;
          }
          mxfp8_quant(values, packed, count);
          int64_t codes = base + row * width + begin;
          int64_t scales =
              base + tileRows * width + row * (width / 32) + begin / 32;
          if (outputStride == 1) {
            std::memcpy(dest + codes, packed, count);
            std::memcpy(dest + scales, packed + count, count / 32);
          } else {
            for (int64_t i = 0; i < count; ++i)
              dest[(codes + i) * outputStride] = packed[i];
            for (int64_t i = 0; i < count / 32; ++i)
              dest[(scales + i) * outputStride] = packed[count + i];
          }
        }
      }
}
