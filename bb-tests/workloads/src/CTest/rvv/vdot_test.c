#include <riscv_vector.h>
#include <stdio.h>

int main() {
  printf("Testing RVV vector dot product with intrinsics\n");

  // Test parameters
  const int n = 8;
  int a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  int b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
  int result = 0;
  // Expected: 1*8 + 2*7 + 3*6 + 4*5 + 5*4 + 6*3 + 7*2 + 8*1
  //         = 8 + 14 + 18 + 20 + 20 + 18 + 14 + 8 = 120
  int expected = 120;

  // Perform vector dot product using RVV intrinsics
  size_t vl = __riscv_vsetvl_e32m1(n); // Set vector length for entire array

  vint32m1_t va = __riscv_vle32_v_i32m1(a, vl);        // Load vector a
  vint32m1_t vb = __riscv_vle32_v_i32m1(b, vl);        // Load vector b
  vint32m1_t vmul = __riscv_vmul_vv_i32m1(va, vb, vl); // Element-wise multiply

  // Manual reduction: store to temp array and sum
  int temp[8];
  __riscv_vse32_v_i32m1(temp, vmul, vl);

  for (int i = 0; i < n; i++) {
    result += temp[i];
  }

  // Verify result
  if (result == expected) {
    printf("Test PASSED - Dot product result: %d\n", result);
    return 0;
  } else {
    printf("Test FAILED - Expected %d, got %d\n", expected, result);
    return 1;
  }
}
