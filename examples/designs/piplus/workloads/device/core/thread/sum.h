#pragma once
#include <stdint.h>

#define SUM_WORDS 8

/* The argument of the thread program, in the thread's TLS: it sums the first
 * `count` words into `total`. */
struct sum_task {
  uint64_t count;
  uint64_t words[SUM_WORDS];
  uint64_t total;
};
