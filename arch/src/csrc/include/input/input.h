#ifndef __INPUT_H__
#define __INPUT_H__

#include "input/datatype.h"
#include "utils/config.h"


fp16_t float_to_fp16(float raw); 
int read_fp16_array(const char *filename, fp16_t array[datain_lines][datain_bandwidth]); 

// void fp16_unpack(fp16 fp16, uint8_t* sign, uint8_t* exponent, uint16_t* fraction);


#endif