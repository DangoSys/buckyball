#ifndef _PARAMS_H
#define _PARAMS_H

#include <limits.h>
#include <stdint.h>

// #define XCUSTOM_ACC 3
// #define DIM 16
// #define MEM_ADDR_LEN 32

// #define SPAD_ADDR_LEN 14
// #define SPAD_BANK_NUM 4
// #define SPAD_BANK_ROWS 4096

// #define ACC_BANK_NUM 1
// #define ACC_BANK_ROWS 1024

typedef int8_t elem_t;
typedef int32_t acc_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;

#define row_align(blocks)                                                      \
  __attribute__((aligned(blocks * DIM * sizeof(elem_t))))

#endif // _PARAMS_H
