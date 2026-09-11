#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern "C" void check_result(int8_t *, int8_t *data, int64_t offset,
                             int64_t rows, int64_t lanes, int64_t stride,
                             int64_t lane_stride) {
  if (rows != 4 || lanes != 16 || stride != 16 || lane_stride != 1)
    exit(1);
  for (int i = 0; i < 64; ++i)
    if (data[offset + i] != 1) {
      printf("mxfp4 unit-scale mismatch index=%d\n", i);
      exit(1);
    }
  printf("mxfp4 unit-scale 2block PASS\n");
}
