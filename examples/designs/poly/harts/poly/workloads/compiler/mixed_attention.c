#include <kernel.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern void _mlir_ciface_compute(struct tensor *, struct tensor *,
                                 struct tensor *);
extern void _mlir_ciface_quantized_views(struct tensor *, struct tensor *,
                                         struct tensor *);
extern void _mlir_ciface_quantized_views_dynamic(struct tensor *,
                                                 struct tensor *,
                                                 struct tensor *);
extern void _mlir_ciface_plain_zero_add(struct tensor *, struct tensor *);
extern void _mlir_ciface_mx_shapes(struct tensor *, struct tensor *);
extern void _mlir_ciface_zero_transpose(struct tensor *, struct tensor *);
extern void _mlir_ciface_quantized_single(struct tensor *, struct tensor *,
                                          struct tensor *);
static float a[2][3][64] __attribute__((aligned(64)));
static float w[2][64][17] __attribute__((aligned(64)));
static float c[2][3][17] __attribute__((aligned(64)));
static float probabilities[64] __attribute__((aligned(64)));
static float values[64][16] __attribute__((aligned(64)));
static float product[16] __attribute__((aligned(64)));

int main(void) {
  struct kernel kernel;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  struct kernel_slot slots[count];
  kernel_init(&kernel, slots, count);
  struct tensor am = {&a[0][0][0], &a[0][0][0], 0, {2, 3, 64}, {192, 64, 1}};
  struct tensor wm = {&w[0][0][0], &w[0][0][0], 0, {2, 64, 17}, {1088, 17, 1}};
  struct tensor cm = {&c[0][0][0], &c[0][0][0], 0, {2, 3, 17}, {51, 17, 1}};
  for (int batch = 0; batch < 2; ++batch) {
    for (int i = 0; i < 3; ++i)
      for (int k = 0; k < 64; ++k)
        a[batch][i][k] = (float)((k + i + batch) % 7 - 3);
    for (int k = 0; k < 64; ++k)
      for (int j = 0; j < 17; ++j)
        w[batch][k][j] = (float)((k + j * 3 + batch) % 5 - 2);
  }
  _mlir_ciface_compute(&am, &wm, &cm);
  for (int batch = 0; batch < 2; ++batch)
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 17; ++j) {
        float expected = 0;
        for (int k = 0; k < 64; ++k)
          expected += a[batch][i][k] * w[batch][k][j];
        if (c[batch][i][j] != expected) {
          printf("Attention mismatch: batch %d row %d column %d: %d != %d\n",
                 batch, i, j, (int)c[batch][i][j], (int)expected);
          return 1;
        }
      }
  for (int k = 0; k < 64; ++k)
    probabilities[k] = k < 32 ? 0x1p-5f : 0x1p-28f;
  for (int k = 0; k < 64; ++k)
    for (int j = 0; j < 16; ++j)
      values[k][j] = 1.0f;
  struct tensor pm = {probabilities, probabilities, 0, {1, 1, 64}, {64, 64, 1}};
  struct tensor vm = {values, values, 0, {1, 64, 16}, {1024, 16, 1}};
  struct tensor out = {product, product, 0, {1, 1, 16}, {16, 16, 1}};
  // The second MXINT8 block adds one ulp; scalar K-order accumulation loses it.
  for (int dynamic = 0; dynamic < 2; ++dynamic) {
    if (dynamic)
      _mlir_ciface_quantized_views_dynamic(&pm, &vm, &out);
    else
      _mlir_ciface_quantized_views(&pm, &vm, &out);
    for (int j = 0; j < 16; ++j) {
      uint32_t bits;
      memcpy(&bits, &product[j], sizeof(bits));
      if (bits != UINT32_C(0x3f800001)) {
        printf(
            "Quantized view mismatch: dynamic %d column %d: %08x != 3f800001\n",
            dynamic, j, bits);
        return 1;
      }
    }
  }
  const int partial_lengths[] = {33, 48, 49};
  for (unsigned i = 0; i < sizeof(partial_lengths) / sizeof(partial_lengths[0]);
       ++i) {
    const int k = partial_lengths[i];
    pm.sizes[2] = pm.strides[0] = pm.strides[1] = k;
    vm.sizes[1] = k;
    vm.strides[0] = k * 16;
    _mlir_ciface_quantized_views_dynamic(&pm, &vm, &out);
    const uint32_t expected =
        k <= 48 ? UINT32_C(0x3f800000) : UINT32_C(0x3f800001);
    for (int j = 0; j < 16; ++j) {
      uint32_t bits;
      memcpy(&bits, &product[j], sizeof(bits));
      if (bits != expected) {
        printf(
            "Quantized partial view mismatch: K %d column %d: %08x != %08x\n",
            k, j, bits, expected);
        return 1;
      }
    }
  }
  for (int k = 0; k < 64; ++k) {
    probabilities[k] = k == 0 ? 1.0f : 16.0f;
    for (int j = 0; j < 16; ++j)
      values[k][j] = k == 0 ? (float)(j + 1) : 1000.0f;
  }
  struct tensor one_p = {probabilities, probabilities, 0, {1, 1, 1}, {1, 1, 1}};
  struct tensor one_v = {values, values, 0, {1, 1, 16}, {16, 16, 1}};
  _mlir_ciface_quantized_single(&one_p, &one_v, &out);
  for (int j = 0; j < 16; ++j)
    if (product[j] != (float)(j + 1)) {
      printf("Static K padding mismatch at column %d\n", j);
      return 1;
    }
  probabilities[0] = -0.0f;
  probabilities[1] = 0.0f;
  probabilities[2] = 1.0f;
  probabilities[3] = -1.0f;
  struct tensor pm4 = {probabilities, probabilities, 0, {1, 1, 4}, {4, 4, 1}};
  struct tensor out4 = {product, product, 0, {1, 1, 4}, {4, 4, 1}};
  _mlir_ciface_plain_zero_add(&pm4, &out4);
  const uint32_t expected[] = {0, 0, 0x3f800000, 0xbf800000};
  if (memcmp(product, expected, sizeof(expected))) {
    puts("Plain FP32 zero addition changed signed-zero behavior");
    return 1;
  }
  const uint32_t source_bits[] = {0x80000000, 0,          0xbf800000,
                                  0x3f800000, 0x7f800000, 0xff800000,
                                  0x7fc01234, 0xffc01234};
  const uint32_t transposed_bits[] = {0,          0x7f800000, 0,
                                      0xff800000, 0xbf800000, 0x7fc00000,
                                      0x3f800000, 0x7fc00000};
  memcpy(probabilities, source_bits, sizeof(source_bits));
  struct tensor source = {
      probabilities, probabilities, 0, {1, 2, 4}, {8, 4, 1}};
  struct tensor target = {product, product, 0, {1, 4, 2}, {8, 2, 1}};
  _mlir_ciface_zero_transpose(&source, &target);
  if (memcmp(product, transposed_bits, sizeof(transposed_bits))) {
    puts("Zero addition across transpose changed values or layout");
    return 1;
  }
  int64_t shape[3];
  struct tensor shape_out = {shape, shape, 0, {1, 1, 3}, {3, 3, 1}};
  const int lengths[] = {1, 31, 32, 33, 63, 64};
  for (unsigned i = 0; i < sizeof(lengths) / sizeof(lengths[0]); ++i) {
    const int k = lengths[i];
    struct tensor input = {values, values, 0, {1, k, 16}, {k * 16, 16, 1}};
    _mlir_ciface_mx_shapes(&input, &shape_out);
    const int blocks = (k + 31) / 32;
    if (shape[0] != blocks * 33 || shape[1] != blocks * 32 * 16 ||
        shape[2] != 16) {
      printf("MXINT8 shape mismatch at length %d\n", k);
      return 1;
    }
  }
  puts("Compiled attention: quantized operands, quantized views, shapes and "
       "signed "
       "zero passed");
  return 0;
}
