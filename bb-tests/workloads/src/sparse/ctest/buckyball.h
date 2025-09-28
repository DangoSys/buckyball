#ifndef BUCKYBALL_H
#define BUCKYBALL_H

#include <stdint.h>

// Data type for matrix elements
typedef int8_t elem_t;
typedef int32_t result_t;

// Utility macros
// #define SPAD_ADDR(bank, row) (((bank) << BANK_ADDR_LEN) | (row))
// #define SP_MATRICES ((BANK_NUM * BANK_ROWS) / DIM)

// Utility functions
void print_matrix(const char *name, elem_t *matrix, int rows, int cols);
void init_matrix(elem_t *matrix, int rows, int cols, int seed);
int compare_matrices(elem_t *a, elem_t *b, int rows, int cols);

#endif
