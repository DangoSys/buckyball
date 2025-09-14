#ifndef _PARAMS_H
#define _PARAMS_H

#include <limits.h>
#include <stdint.h>

typedef int8_t elem_t;
typedef int32_t acc_t;
static const elem_t elem_t_max = 127;
static const elem_t elem_t_min = -128;

#define row_align(blocks)                                                      \
  __attribute__((aligned(blocks * DIM * sizeof(elem_t))))

#endif // _PARAMS_H
