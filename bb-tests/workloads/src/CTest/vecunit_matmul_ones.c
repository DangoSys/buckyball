#include "buckyball.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t input_matrix_b[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t expected_matrix[DIM * DIM] __attribute__((aligned(64)));
#define BANK 512
#define OP1_ADDR 0
#define OP2_ADDR (BANK + DIM)
#define WR_ADDR (DIM + 2 * BANK)

void hw_matmul(const char* test_name, elem_t* a, elem_t* b, result_t* c, int size) {
    static elem_t a_transposed[DIM * DIM] __attribute__((aligned(64)));
    transpose_u8_matrix(a, a_transposed, size, size);
    bb_mvin((uintptr_t)a_transposed, OP1_ADDR, size);
    bb_mvin((uintptr_t)b, OP2_ADDR, size);
    bb_mvin((uintptr_t)c, WR_ADDR, size << 2);
    bb_fence();
    bb_mul_warp16(OP1_ADDR, OP2_ADDR, WR_ADDR, size);
    bb_fence();
    bb_mvout((uintptr_t)c, WR_ADDR, size << 2);
    bb_fence();
}

int run_test(const char* test_name, elem_t* a, elem_t* b, int size) {
    clear_u32_matrix(output_matrix, DIM, DIM);
    cpu_matmul(a, b, expected_matrix, size, size, size);
    hw_matmul(test_name, a, b, output_matrix, size);
    if (compare_u32_matrices(output_matrix, expected_matrix, size, size)) {
        printf("Test %s PASSED\n", test_name);
        return 1;
    } else {
        printf("Test %s FAILED\n", test_name);
        return 0;
    }
}

int test_ones() {
    init_ones_matrix(input_matrix_a, DIM, DIM);
    init_ones_matrix(input_matrix_b, DIM, DIM);
    return run_test("All-ones matrices", input_matrix_a, input_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE 
    multicore(MULTICORE);
#endif
    int passed = test_ones();
    if (passed) {
        printf("vecunit_matmul_ones test PASSED\n");
    } else {
        printf("vecunit_matmul_ones test FAILED\n");
    }
#ifdef MULTICORE 
    exit(0);
#endif
} 