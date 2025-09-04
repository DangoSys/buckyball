#include "buckyball.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static elem_t input_matrix_a[DIM * DIM] __attribute__((aligned(64)));
static elem_t output_matrix_b[DIM * 1024] __attribute__((aligned(64)));

#define BANK 512
#define OP1_ADDR 0
#define OP2_ADDR DIM + BANK


void hw_im2col(const char* test_name, elem_t* a, elem_t* b, int size) {
    bb_mvin((uintptr_t)a, OP1_ADDR, size);
    bb_fence();
    bb_im2col(OP1_ADDR, OP2_ADDR, 2, size);
    bb_fence();
}

int run_test(const char* test_name, elem_t* a, elem_t* b, int size) {
    hw_im2col(test_name, a, b, size);
    return 1;
}

int test_im2col() {
    init_sequence_matrix(input_matrix_a, DIM, DIM);
    return run_test("Im2col", input_matrix_a, output_matrix_b, DIM);
}

int main() {
#ifdef MULTICORE 
    multicore(MULTICORE);
#endif
    int passed = test_im2col();
    if (passed) {
        printf("Im2col test PASSED\n");
    } else {
        printf("Im2col test FAILED\n");
    }
#ifdef MULTICORE 
    exit(0);
#endif
} 