#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define IFMAP_SIZE 64
#define WEIGHT_SIZE 16
#define OFMAP_SIZE 16

static elem_t input_feature_map[IFMAP_SIZE] __attribute__((aligned(64)));
static elem_t weights[WEIGHT_SIZE] __attribute__((aligned(64)));
static elem_t output_feature_map[OFMAP_SIZE] __attribute__((aligned(64)));

// CPU reference computation for simple convolution
int conv_cpu_reference(elem_t *ifmap, elem_t *weight, elem_t *ofmap, int in_h,
                       int in_w, int kernel_h, int kernel_w) {
  // Simplified 2D convolution: assume stride=1, pad=0
  int out_h = in_h - kernel_h + 1;
  int out_w = in_w - kernel_w + 1;

  for (int oh = 0; oh < out_h; oh++) {
    for (int ow = 0; ow < out_w; ow++) {
      int32_t sum = 0;
      for (int kh = 0; kh < kernel_h; kh++) {
        for (int kw = 0; kw < kernel_w; kw++) {
          int ih = oh + kh;
          int iw = ow + kw;
          int ifmap_idx = ih * in_w + iw;
          int weight_idx = kh * kernel_w + kw;
          sum += (int32_t)ifmap[ifmap_idx] * (int32_t)weight[weight_idx];
        }
      }
      // Clamp to int8_t range
      if (sum > 127) {
        ofmap[oh * out_w + ow] = 127;
      } else if (sum < -128) {
        ofmap[oh * out_w + ow] = -128;
      } else {
        ofmap[oh * out_w + ow] = (elem_t)sum;
      }
    }
  }
  return 1;
}

void hw_conv(const char *test_name, elem_t *ifmap, elem_t *weight,
             elem_t *ofmap, int in_h, int in_w, int kernel_h, int kernel_w) {
  // Input feature map in spad bank 0, weights in spad bank 1, output in spad
  // bank 2
  uint32_t ifmap_addr = spad_addr(0, 0);
  uint32_t weight_addr = spad_addr(1, 0);
  uint32_t ofmap_addr = spad_addr(2, 0);

  // Move input feature map into scratchpad
  bb_mvin((uintptr_t)ifmap, ifmap_addr, IFMAP_SIZE, 1);
  bb_fence();

  // Move weights into scratchpad
  bb_mvin((uintptr_t)weight, weight_addr, WEIGHT_SIZE, 1);
  bb_fence();

  // Call CONV instruction
  // iter is the number of iterations (simplified: use 1 for now)
  uint32_t iter = 1;
  bb_conv(ifmap_addr, weight_addr, ofmap_addr, iter, in_h, in_w, kernel_h,
          kernel_w);
  bb_fence();

  // Result will be moved back in run_test for verification
}

int run_test(const char *test_name, elem_t *ifmap, elem_t *weight,
             elem_t *ofmap, int in_h, int in_w, int kernel_h, int kernel_w) {
  // CPU reference computation
  conv_cpu_reference(ifmap, weight, ofmap, in_h, in_w, kernel_h, kernel_w);

  // Hardware computation
  hw_conv(test_name, ifmap, weight, ofmap, in_h, in_w, kernel_h, kernel_w);

  // Move result back from scratchpad for verification
  uint32_t ofmap_addr = spad_addr(2, 0);
  bb_mvout((uintptr_t)output_feature_map, ofmap_addr, OFMAP_SIZE, 1);
  bb_fence();

  // Verify results
  int out_h = in_h - kernel_h + 1;
  int out_w = in_w - kernel_w + 1;
  int passed = 1;
  for (int i = 0; i < out_h; i++) {
    for (int j = 0; j < out_w; j++) {
      int idx = i * out_w + j;
      if (output_feature_map[idx] != ofmap[idx]) {
        printf("Mismatch at [%d][%d]: expected %d, got %d\n", i, j, ofmap[idx],
               output_feature_map[idx]);
        passed = 0;
      }
    }
  }

  return passed;
}

int test_conv(int seed) {
  // Initialize input feature map with random values (8x8 image)
  int in_h = 8;
  int in_w = 8;
  for (int i = 0; i < IFMAP_SIZE; i++) {
    input_feature_map[i] = (elem_t)(rand() % 256 - 128);
  }

  // Initialize weights with random values (3x3 kernel)
  int kernel_h = 3;
  int kernel_w = 3;
  for (int i = 0; i < WEIGHT_SIZE; i++) {
    weights[i] = (elem_t)(rand() % 256 - 128);
  }

  // Run hardware test with verification
  return run_test("CONV", input_feature_map, weights, output_feature_map, in_h,
                  in_w, kernel_h, kernel_w);
}

int main() {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_conv(5);
  if (passed) {
    printf("CONV test PASSED!!!!\n");
  } else {
    printf("CONV test FAILED\n");
  }
  return (!passed);

#ifdef MULTICORE
  exit(0);
#endif
}
