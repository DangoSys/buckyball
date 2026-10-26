#pragma once
#include <stdint.h>
struct ant_corelink_reduce_args {
  uint64_t phase, input, output;
};
void ant_corelink_reduce_phase(unsigned core,
                               const struct ant_corelink_reduce_args *args);
void ant_corelink_reduce_prepare(unsigned core, float *input, float *output);
