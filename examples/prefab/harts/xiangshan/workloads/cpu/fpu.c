// Single- and double-precision FPU: add, mul, div, sqrt, fma, conversions.
#include <stdint.h>
#include <stdio.h>

static double sqrt_d(double x) {
  double r;
  asm volatile("fsqrt.d %0, %1" : "=f"(r) : "f"(x));
  return r;
}

static float sqrt_s(float x) {
  float r;
  asm volatile("fsqrt.s %0, %1" : "=f"(r) : "f"(x));
  return r;
}

int main(void) {
  volatile double x = 1.5, y = -2.25;
  volatile float f = 3.0f, g = 0.5f;
  double d = x * y + x / y;
  if (d != -3.375 + (-0.6666666666666666)) {
    printf("fpu double FAIL\n");
    return 1;
  }
  if (sqrt_d(x * 6.0) != 3.0 || __builtin_fma(x, y, 1.0) != -2.375) {
    printf("fpu sqrt/fma FAIL\n");
    return 1;
  }
  if (f * g + f != 4.5f || sqrt_s(f * f) != 3.0f) {
    printf("fpu float FAIL\n");
    return 1;
  }
  volatile double big = -12345.75;
  if ((int64_t)big != -12345 || (double)(int64_t)-7 != -7.0 ||
      (float)x != 1.5f) {
    printf("fpu convert FAIL\n");
    return 1;
  }
  printf("fpu PASS\n");
  return 0;
}
