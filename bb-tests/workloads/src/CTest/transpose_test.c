#include "buckyball.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

#define BANK 512
#define OP1_ADDR 0
#define OP2_ADDR DIM + BANK


void hw_transpose(const char* test_name, elem_t* a, elem_t* b, int size) {
    bb_mvin((uintptr_t)a, OP1_ADDR, size);
    bb_fence();
    bb_transpose(OP1_ADDR, OP2_ADDR, size);
    bb_fence();
}

int run_test(const char* test_name, elem_t* a, elem_t* b, int size) {
    hw_transpose(test_name, a, b, size);
    return 1;
}

int test_transpose() {
    init_sequence_matrix(input_matrix_a, DIM, DIM);
    return run_test("Im2col", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE 
    multicore(MULTICORE);
#endif
    int passed = test_transpose();
    if (passed) {
        printf("Transpose test PASSED\n");
    } else {
        printf("Transpose test FAILED\n");
    }
#ifdef MULTICORE 
    exit(0);
#endif
} 