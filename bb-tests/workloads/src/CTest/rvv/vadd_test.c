#include <riscv_vector.h>
#include <stdio.h>

int main() {
  printf("Testing RVV vector addition with intrinsics\n");

  // Test parameters
  const int n = 16;
  int a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  int b[16] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  int c[16] = {0};
  int expected[16] = {17, 17, 17, 17, 17, 17, 17, 17,
                      17, 17, 17, 17, 17, 17, 17, 17};

  // Perform vector addition using RVV intrinsics
  size_t vl;
  for (size_t i = 0; i < n;) {
    vl = __riscv_vsetvl_e32m1(n - i);                  // Set vector length
    vint32m1_t va = __riscv_vle32_v_i32m1(&a[i], vl);  // Load vector a
    vint32m1_t vb = __riscv_vle32_v_i32m1(&b[i], vl);  // Load vector b
    vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, vl); // Vector add
    __riscv_vse32_v_i32m1(&c[i], vc, vl);              // Store result
    i += vl;
  }

  // Verify results
  int passed = 1;
  for (int i = 0; i < n; i++) {
    if (c[i] != expected[i]) {
      printf("FAILED at index %d: expected %d, got %d\n", i, expected[i], c[i]);
      passed = 0;
    }
  }

  if (passed) {
    printf("Test PASSED - Vector addition works correctly!\n");
  }

  return passed ? 0 : 1;
}
