#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <bbhw/mem/mem.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BLOCK_ELEMS 16
#define NUM_BLOCKS 16

#define BANK_WORD_BYTES 16 // 128-bit bank word
#define ELEMS_PER_BANK_WORD 4
#define WORDS_PER_BLOCK 4
#define OUT_BYTES_PER_BLOCK 16 // one packed MX block = one 128-bit word
#define TOTAL_INPUT_WORDS (NUM_BLOCKS * BLOCK_ELEMS)
#define TOTAL_OUT_BYTES (NUM_BLOCKS * OUT_BYTES_PER_BLOCK)

static uint32_t input_bits[TOTAL_INPUT_WORDS] __attribute__((aligned(64)));
static uint8_t expected_output[TOTAL_OUT_BYTES] __attribute__((aligned(64)));
static uint8_t output_buffer[TOTAL_OUT_BYTES] __attribute__((aligned(64)));

static inline uint32_t fp_sign(uint32_t x) { return (x >> 31) & 0x1u; }
static inline uint32_t fp_exp(uint32_t x) { return (x >> 23) & 0xffu; }
static inline uint32_t fp_frac(uint32_t x) { return x & 0x7fffffu; }

static inline int is_zero_u32(uint32_t x) {
  return (fp_exp(x) == 0u) && (fp_frac(x) == 0u);
}

static inline int is_subnormal_u32(uint32_t x) {
  return (fp_exp(x) == 0u) && (fp_frac(x) != 0u);
}

static inline int is_special_u32(uint32_t x) { return fp_exp(x) == 0xffu; }

static inline uint8_t normal_exp_or_zero(uint32_t x) {
  if (is_zero_u32(x) || is_subnormal_u32(x) || is_special_u32(x)) {
    return 0;
  }
  return (uint8_t)fp_exp(x);
}

static uint8_t quantize_mag4(uint32_t x, uint8_t shared_exp) {
  uint32_t exp = fp_exp(x);
  uint32_t frac = fp_frac(x);

  if (is_zero_u32(x) || is_subnormal_u32(x))
    return 0;
  if (is_special_u32(x))
    return 15;
  if (exp > shared_exp)
    return 15;

  uint64_t sig24 = (1ull << 23) | frac;
  uint32_t shift_amt = (20u + (uint32_t)shared_exp - exp) & 0x3fu;
  uint64_t shifted = sig24 >> shift_amt;

  if (shifted >= 15u)
    return 15;
  return (uint8_t)(shifted & 0xfu);
}

static void pack_mx6_block(const uint32_t *block_in, uint8_t *block_out_16B) {
  uint8_t exps[BLOCK_ELEMS];
  uint8_t global_exp = 0;
  uint8_t micro_byte = 0;
  uint8_t payloads[BLOCK_ELEMS];

  memset(block_out_16B, 0, OUT_BYTES_PER_BLOCK);

  for (int i = 0; i < BLOCK_ELEMS; ++i) {
    exps[i] = normal_exp_or_zero(block_in[i]);
    if (exps[i] > global_exp)
      global_exp = exps[i];
  }

  for (int p = 0; p < BLOCK_ELEMS / 2; ++p) {
    uint8_t e0 = exps[2 * p];
    uint8_t e1 = exps[2 * p + 1];
    uint8_t pair_max = (e0 > e1) ? e0 : e1;

    uint8_t micro = 0;
    if ((global_exp != 0u) &&
        ((uint16_t)pair_max + 1u <= (uint16_t)global_exp)) {
      micro = 1;
    }
    if (micro)
      micro_byte |= (uint8_t)(1u << p);

    uint8_t local_exp = micro ? (uint8_t)(global_exp - 1u) : global_exp;

    for (int k = 0; k < 2; ++k) {
      int idx = 2 * p + k;
      uint8_t sign = (uint8_t)fp_sign(block_in[idx]);
      uint8_t mag = quantize_mag4(block_in[idx], local_exp);
      payloads[idx] = (uint8_t)((sign << 4) | mag);
    }
  }

  block_out_16B[0] = global_exp;
  block_out_16B[1] = micro_byte;

  for (int i = 0; i < BLOCK_ELEMS; ++i) {
    uint32_t bit_pos = (uint32_t)(i * 5);
    uint32_t byte_pos = bit_pos / 8;
    uint32_t bit_off = bit_pos % 8;
    uint16_t val = (uint16_t)(payloads[i] & 0x1fu);

    block_out_16B[2 + byte_pos] |= (uint8_t)(val << bit_off);
    if (bit_off > 3) {
      block_out_16B[2 + byte_pos + 1] |= (uint8_t)(val >> (8 - bit_off));
    }
  }
}

static void init_input_bits(void) {
  static const uint32_t base_block[BLOCK_ELEMS] = {
      0xBF000000u, // -0.5
      0x3F400000u, //  0.75
      0xBF800000u, // -1.0
      0x3FC00000u, //  1.5
      0xC0000000u, // -2.0
      0x40400000u, //  3.0
      0xC0800000u, // -4.0
      0x40C00000u, //  6.0
      0xC1000000u, // -8.0
      0x41400000u, // 12.0
      0xC1800000u, // -16.0
      0x41C00000u, // 24.0
      0xC2000000u, // -32.0
      0x42400000u, // 48.0
      0xC2800000u, // -64.0
      0x42C00000u  // 96.0
  };

  for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
    for (int i = 0; i < BLOCK_ELEMS; ++i) {
      input_bits[blk * BLOCK_ELEMS + i] = base_block[i];
    }
  }
}

static void build_expected_output(void) {
  memset(expected_output, 0, sizeof(expected_output));
  for (int blk = 0; blk < NUM_BLOCKS; ++blk) {
    pack_mx6_block(&input_bits[blk * BLOCK_ELEMS],
                   &expected_output[blk * OUT_BYTES_PER_BLOCK]);
  }
}

static int compare_bytes(const uint8_t *got, const uint8_t *exp, int n) {
  for (int i = 0; i < n; ++i) {
    if (got[i] != exp[i]) {
      printf("Mismatch at index %d: Expected %u, Got %u\n", i, (unsigned)exp[i],
             (unsigned)got[i]);
      return 0;
    }
  }
  return 1;
}

static void hw_mxfp(const char *test_name, uint32_t *src_bits,
                    uint8_t *dst_bytes) {
  (void)test_name;

  uint32_t op1_bank_id = 0;
  uint32_t wr_bank_id = 1;

  bb_mem_alloc(op1_bank_id, 1, 1);
  bb_mem_alloc(wr_bank_id, 1, 1);

  bb_mvin((uintptr_t)src_bits, op1_bank_id, NUM_BLOCKS * WORDS_PER_BLOCK, 1);
  bb_mxfp(op1_bank_id, wr_bank_id, NUM_BLOCKS);
  bb_mvout((uintptr_t)dst_bytes, wr_bank_id, NUM_BLOCKS, 1);
  bb_fence();
}

static int run_test(const char *test_name) {
  memset(output_buffer, 0, sizeof(output_buffer));

  init_input_bits();
  build_expected_output();
  hw_mxfp(test_name, input_bits, output_buffer);

  if (compare_bytes(output_buffer, expected_output, TOTAL_OUT_BYTES)) {
    printf("%s compare PASSED\n", test_name);
    return 1;
  } else {
    printf("%s compare FAILED\n", test_name);
    return 0;
  }
}

int test_mxfp(int seed) {
  (void)seed;
  return run_test("MXFP");
}

int main(void) {
#ifdef MULTICORE
  multicore(MULTICORE);
#endif

  int passed = test_mxfp(5);
  if (passed) {
    printf("MXFP hardware test PASSED!\n");
  } else {
    printf("MXFP hardware test FAILED!\n");
  }

#ifdef MULTICORE
  exit(0);
#endif

  return !passed;
}
