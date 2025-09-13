#ifndef BUCKYBALL_H
#define BUCKYBALL_H

#include <stdint.h>

// #define DIM 16
// #define SPAD_ADDR_LEN 14
// #define MEM_ADDR_LEN 32
// #define BANK_NUM 4
// #define BANK_ROWS 4096
// #define RF_BANKS 2
// #define BANK_ADDR_LEN 12
// #define BANK_SEL_LEN 2
// #define RF_ADDR_LEN 4
// #define XCUSTOM_ACC 3

// Data type for matrix elements
typedef int8_t elem_t;

// Utility macros
// #define SPAD_ADDR(bank, row) (((bank) << BANK_ADDR_LEN) | (row))
// #define SP_MATRICES ((BANK_NUM * BANK_ROWS) / DIM)

// Utility functions
void print_matrix(const char *name, elem_t *matrix, int rows, int cols);
void init_matrix(elem_t *matrix, int rows, int cols, int seed);
int compare_matrices(elem_t *a, elem_t *b, int rows, int cols);

#endif
