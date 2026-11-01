#pragma once
#include <stdint.h>

/* The argument of a test's thread program, in the thread's TLS: the DDR
 * addresses of the FP32 input bank image and of the output bank image. */
struct mxint8quantize_task {
  uint64_t input;
  uint64_t output;
};
