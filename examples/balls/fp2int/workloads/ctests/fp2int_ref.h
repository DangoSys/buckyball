#ifndef FP2INT_REF_H
#define FP2INT_REF_H

#include <stdint.h>

static uint32_t fp32_mul(uint32_t a, uint32_t b) {
  uint32_t a_sign = (a >> 31) & 1;
  uint32_t b_sign = (b >> 31) & 1;
  uint32_t a_exp = (a >> 23) & 0xff;
  uint32_t b_exp = (b >> 23) & 0xff;
  uint32_t a_frac = a & 0x7fffff;
  uint32_t b_frac = b & 0x7fffff;
  uint64_t a_mant = (1ull << 23) | a_frac;
  uint64_t b_mant = (1ull << 23) | b_frac;
  int a_zero = a_exp == 0 && a_frac == 0;
  int b_zero = b_exp == 0 && b_frac == 0;
  uint64_t prod = a_mant * b_mant;
  uint32_t sig, exp_adjust;
  int guard, roundb, sticky;
  if ((prod >> 47) & 1) {
    sig = (uint32_t)(prod >> 24);
    guard = ((prod >> 23) & 1) != 0;
    roundb = ((prod >> 22) & 1) != 0;
    sticky = (prod & ((1ull << 22) - 1)) != 0;
    exp_adjust = 1;
  } else {
    sig = (uint32_t)(prod >> 23);
    guard = ((prod >> 22) & 1) != 0;
    roundb = ((prod >> 21) & 1) != 0;
    sticky = (prod & ((1ull << 21) - 1)) != 0;
    exp_adjust = 0;
  }
  int increment = guard && (roundb || sticky || (sig & 1) != 0);
  uint64_t rounded = (uint64_t)sig + (uint64_t)increment;
  uint32_t carry = (uint32_t)((rounded >> 24) & 1);
  uint32_t final_sig = (uint32_t)(carry != 0 ? (rounded >> 1) : rounded);
  uint32_t exp_wide = (a_exp + b_exp + exp_adjust + carry - 127) & 0x3ff;

  if (a_zero || b_zero)
    return 0;
  if (exp_wide & 0x200)
    return 0;
  if (exp_wide & 0x100)
    return ((a_sign ^ b_sign) << 31) | (0xffu << 23);
  return ((a_sign ^ b_sign) << 31) | ((exp_wide & 0xff) << 23) |
         (final_sig & 0x7fffff);
}

static int32_t fp32_to_i32(uint32_t fp) {
  int sign = ((fp >> 31) & 1) != 0;
  int32_t exponent = (int32_t)((fp >> 23) & 0xff);
  uint32_t frac = fp & 0x7fffff;
  uint32_t mantissa = (1u << 23) | frac;
  int is_zero = exponent == 0 && frac == 0;
  int32_t exp_val = exponent - 127;

  if (exponent == 0xff && frac != 0)
    return 0;
  if (is_zero)
    return 0;

  uint64_t magnitude;
  if (exp_val >= 31) {
    magnitude = 0x80000000ull;
  } else if (exp_val >= 23) {
    magnitude = (uint64_t)mantissa << (exp_val - 23);
  } else if (exp_val >= -1) {
    uint32_t right_shift = (uint32_t)(23 - exp_val);
    uint64_t truncated = (uint64_t)mantissa >> right_shift;
    uint64_t half = 1ull << (right_shift - 1);
    uint64_t remainder = (uint64_t)mantissa & ((1ull << right_shift) - 1);
    int round_up =
        remainder > half || (remainder == half && (truncated & 1) != 0);
    magnitude = truncated + (uint64_t)round_up;
  } else {
    magnitude = 0;
  }

  if (sign) {
    if (magnitude >= 0x80000000ull)
      return (int32_t)0x80000000;
    return -(int32_t)magnitude;
  }
  if (magnitude > 0x7fffffffull)
    return 0x7fffffff;
  return (int32_t)magnitude;
}

static int32_t fp2int_i32(uint32_t fp_bits, uint32_t scale_bits) {
  return fp32_to_i32(fp32_mul(fp_bits, scale_bits));
}

static int8_t fp2int_i8(uint32_t fp_bits, uint32_t scale_bits) {
  int32_t v = fp2int_i32(fp_bits, scale_bits);
  if (v > 127)
    return 127;
  if (v < -128)
    return -128;
  return (int8_t)v;
}

static float bits_to_f32(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } x = {.u = bits};
  return x.f;
}

static uint32_t f32_to_bits(float f) {
  union {
    float f;
    uint32_t u;
  } x = {.f = f};
  return x.u;
}

#endif
