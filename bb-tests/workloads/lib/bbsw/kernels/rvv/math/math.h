#pragma once
#include <riscv_vector.h>

extern "C" vfloat32m1_t rvv_exp(vfloat32m1_t value, size_t vl);
extern "C" vfloat32m1_t rvv_sin(vfloat32m1_t value, size_t vl);
extern "C" vfloat32m1_t rvv_cos(vfloat32m1_t value, size_t vl);
