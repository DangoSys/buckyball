#include <CRunnerUtils.h>
#include <algorithm>
#include <bbhw/isa/isa.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <isa/gemmini.h>

extern "C" void _mlir_ciface_gemmini_conv(StridedMemRefType<float, 4> *input,
                                          StridedMemRefType<int8_t, 2> *weight,
                                          StridedMemRefType<float, 1> *bias,
                                          StridedMemRefType<float, 4> *output,
                                          int64_t kh, int64_t kw,
                                          int64_t stride, int64_t padding,
                                          float inputScale, float weightScale) {
  const int64_t batches = input->sizes[0], channels = input->sizes[1],
                height = input->sizes[2], width = input->sizes[3];
  const int64_t oh = output->sizes[2], ow = output->sizes[3],
                n = weight->sizes[1], k = channels * kh * kw,
                m = batches * oh * ow;
  if (k != weight->sizes[0] || n != bias->sizes[0] ||
      output->sizes[0] != batches || output->sizes[1] != n ||
      oh != (height + 2 * padding - kh) / stride + 1 ||
      ow != (width + 2 * padding - kw) / stride + 1 ||
      k > INT32_MAX / (128 * 128) || !std::isfinite(inputScale) ||
      inputScale <= 0 || !std::isfinite(weightScale) || weightScale <= 0) {
    fputs("invalid Gemmini convolution shape/scales\n", stderr);
    abort();
  }
  int8_t a[256] __attribute__((aligned(64))),
      b[256] __attribute__((aligned(64)));
  int32_t c[256] __attribute__((aligned(64)));
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 4);
  bb_mset_clear(2, 1, 1);
  bb_gemmini_config(1, 0, 0, 0, 0);
  for (int64_t row = 0; row < m; row += 16)
    for (int64_t col = 0; col < n; col += 16) {
      for (int64_t inner = 0; inner < k; inner += 16) {
        for (int i = 0; i < 16; ++i)
          for (int j = 0; j < 16; ++j) {
            int64_t index = row + i, channel = (inner + j) % channels,
                    ky = (inner + j) / (kw * channels),
                    kx = ((inner + j) / channels) % kw;
            int64_t batch = index / (oh * ow),
                    y = (index / ow) % oh * stride + ky - padding,
                    x = index % ow * stride + kx - padding;
            float value = 0;
            if (index < m && inner + j < k && y >= 0 && y < height && x >= 0 &&
                x < width)
              value =
                  input->data[input->offset + batch * input->strides[0] +
                              channel * input->strides[1] +
                              y * input->strides[2] + x * input->strides[3]];
            if (!std::isfinite(value)) {
              fputs("non-finite Gemmini activation\n", stderr);
              abort();
            }
            a[i * 16 + j] = int8_t(
                std::clamp(std::nearbyint(value / inputScale), -128.f, 127.f));
            b[i * 16 + j] =
                inner + i < k && col + j < n
                    ? weight->data[weight->offset +
                                   (inner + i) * weight->strides[0] +
                                   (col + j) * weight->strides[1]]
                    : 0;
          }
        bb_mvin((uintptr_t)a, 1, 16, 1);
        bb_mvin((uintptr_t)b, 0, 16, 1);
        if (inner == 0) {
          bb_gemmini_preload(0, 3, 16, 0, 0);
          bb_gemmini_compute_preloaded(1, 2, 3, 16, 0, 0, 0);
        } else
          bb_gemmini_compute_accumulated(1, 0, 3, 16, 0, 0, 0);
      }
      bb_mvout((uintptr_t)c, 3, 16, 1);
      bb_fence();
      for (int i = 0; i < 16 && row + i < m; ++i)
        for (int j = 0; j < 16 && col + j < n; ++j) {
          int64_t index = row + i, batch = index / (oh * ow),
                  y = (index / ow) % oh, x = index % ow;
          output->data[output->offset + batch * output->strides[0] +
                       (col + j) * output->strides[1] + y * output->strides[2] +
                       x * output->strides[3]] =
              float(c[i * 16 + j]) * (inputScale * weightScale) +
              bias->data[bias->offset + (col + j) * bias->strides[0]];
        }
    }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
}
