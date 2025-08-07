#include "buckyball.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define BANK 512
#define OP1_ADDR 0
#define OP2_ADDR (BANK + DIM)
#define WR_ADDR (DIM + 2 * BANK)

void init_u8_random_matrix(elem_t* matrix, int rows, int cols, int seed) {
    srand(seed);  
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 128;  
    }
}

void init_u32_random_matrix(result_t* matrix, int rows, int cols, int seed) {
    srand(seed);  
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = rand() % 256;  
    }
}
int compare_u8_matrices(elem_t* a, elem_t* b, int rows, int cols) {
   for (int i = 0; i < rows * cols; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
            //print_matrix("Expected", b, 1, cols);
            //print_matrix("Actual", a, 1, cols);
            return 0;
        }
    }
    return 1;
}
int compare_u32_matrices(result_t* a, result_t* b, int rows, int cols) {
    for (int i = 0; i <= rows * cols - 1; i++) {
        if (a[i] != b[i]) {
            printf("Mismatch at index %d: expected %d, got %d\n", i, b[i], a[i]);
            return 0;
        }
    }
    return 1;
}

void print_u32_matrix(const char* name, result_t* matrix, int rows, int cols) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}
void print_u8_matrix(const char* name, elem_t* matrix, int rows, int cols) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%4d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}
void clear_u32_matrix(result_t* matrix, int rows, int cols) {
    memset(matrix, 0, rows * cols * sizeof(result_t));
}
void clear_u8_matrix(elem_t* matrix, int rows, int cols) {
    memset(matrix, 0, rows * cols * sizeof(elem_t));
}
void init_ones_matrix(elem_t* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = 1;
    }
}

void init_identity_matrix(elem_t* matrix, int size) {
    clear_u8_matrix(matrix, size, size);
    for (int i = 0; i < size; i++) {
        matrix[i * size + i] = 1;
    }
}

void init_row_vector(elem_t* matrix, int cols, elem_t value) {
    clear_u8_matrix(matrix, DIM, DIM);
    for (int j = 0; j < cols; j++) {
        matrix[j] = value;
    }
}

void init_col_vector(elem_t* matrix, int rows, elem_t value) {
    clear_u8_matrix(matrix, DIM, DIM);
    for (int i = 0; i < rows; i++) {
        matrix[i * DIM] = value;
    }
}

void init_random_matrix(elem_t* matrix, int rows, int cols, int seed) {
    srand(seed);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (rand() % 5); // 0-4的随机数
        }
    }
}

void init_bbfp_random_matrix(elem_t* matrix, int rows, int cols, int seed) {
    srand(seed);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i * cols + j] = (rand() % 16); 
        }
    }
}

// 转置矩阵
void transpose_u8_matrix(elem_t* src, elem_t* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}
void transpose_u32_matrix(result_t* src, result_t* dst, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

// CPU矩阵乘法（用于生成期望结果）
void cpu_matmul(elem_t* a, elem_t* b, result_t* c, int rows, int cols, int inner) {
    clear_u32_matrix(c, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < inner; k++) {
                c[i * cols + j] += a[i * inner + k] * b[k * cols + j];
            }
        }
    }
}
