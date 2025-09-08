#ifndef BUCKYBALL_PARAMS_H
#define BUCKYBALL_PARAMS_H

#include <stdint.h>
#include <limits.h>

#define XCUSTOM_ACC 3
#define DIM 16
#define MEM_ADDR_LEN 32
#define SPAD_ADDR_LEN 14
#define BANK_NUM 4
#define BANK_ROWS 4096

typedef int8_t elem_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))

#endif // BUCKYBALL_PARAMS_H
