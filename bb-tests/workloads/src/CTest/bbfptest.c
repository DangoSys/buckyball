#include "buckyball.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



// Test matrices
static elem_t input_matrix[DIM * DIM] __attribute__((aligned(64)));
static elem_t weight_matrix[DIM * DIM] __attribute__((aligned(64)));
static result_t output_matrix[DIM * DIM] __attribute__((aligned(64)));


#define BANK 512
// Utility function     
void print_result_matrix(const char* name, result_t* matrix, int rows, int cols) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", (int32_t)matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void init_matrixv2(elem_t* matrix, int rows, int cols, int seed,int value) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = value;  
    }
}

int compare_matrices(result_t* a, result_t* b, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        if (a[i] != b[i]) {
            return 0;  // Matrices are different
        }
    }
    return 1;  // Matrices are the same
}

#define OP1_ADDR 0
#define OP2_ADDR (BANK + DIM)
#define WR_ADDR (DIM + 2 * BANK)
int main() {
#ifdef MULTICORE 
    multicore(MULTICORE);  // Only allow specified hart to continue
#endif
    
    // Initialize input matrix
    init_matrixv2(input_matrix, 16, 16, 42, 3);
    init_matrixv2(weight_matrix, 16, 16, 42, 2);
    // Clear output matrix
    memset(output_matrix, 0, sizeof(output_matrix));
    
    //print_matrix("Input", input_matrix, DIM, DIM);
    
    // Move input to scratchpad
    bb_mvin((uintptr_t)input_matrix, OP1_ADDR, DIM);
    bb_mvin((uintptr_t)weight_matrix, OP2_ADDR, DIM);
    bb_mvin((uintptr_t)output_matrix, WR_ADDR, DIM << 2);
    printf("Perform Matmul\n");
    bb_bbfp_mul(OP1_ADDR, OP2_ADDR, WR_ADDR, DIM);
   
    printf("change");
    bb_matmul_ws(WR_ADDR, OP2_ADDR, WR_ADDR, 16);
    init_matrixv2(input_matrix, 16, 16, 42, 4);
    bb_matmul_ws(WR_ADDR, OP2_ADDR, WR_ADDR, 16);
    printf("Matmul Done\n");
    bb_mvout(((uintptr_t)output_matrix), WR_ADDR, DIM << 2);

    print_result_matrix("Output", output_matrix, DIM, DIM);
    
#ifdef MULTICORE 
   exit(0);
#endif

}
