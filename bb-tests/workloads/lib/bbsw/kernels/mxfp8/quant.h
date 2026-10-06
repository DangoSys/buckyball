#pragma once
#include <stdint.h>

extern "C" void mxfp8_quant(const uint32_t *input, uint8_t *output,
                            uint32_t count);
