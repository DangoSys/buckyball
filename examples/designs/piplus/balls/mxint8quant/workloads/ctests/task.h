#pragma once
#include <stdint.h>

/* The argument of a test's thread program, in the thread's TLS: the DDR
 * addresses of the FP32 input image and of the packed output image. */
struct ball_task {
  uint64_t input;
  uint64_t output;
};
