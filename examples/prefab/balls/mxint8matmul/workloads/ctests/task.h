#pragma once
#include <stdint.h>

/* The argument of a test's thread program, in the thread's TLS: the DDR
 * addresses of the A and B bank images and of the C image (the old C of an
 * accumulating test, overwritten with the C bank). */
struct mxint8matmul_task {
  uint64_t a;
  uint64_t b;
  uint64_t c;
};
