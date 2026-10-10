#pragma once
#include <stdint.h>
struct ant_tlink_reduce_args {
  uint64_t phase, input, output;
};
void ant_tlink_reduce_phase(unsigned tile,
                            const struct ant_tlink_reduce_args *args);
void ant_tlink_reduce_prepare(unsigned tile, float *input, float *output);
