#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" void check_result(int32_t *, int32_t *values, int64_t offset,
                             int64_t rows, int64_t columns, int64_t rs,
                             int64_t cs, int64_t *, int64_t *indices,
                             int64_t index_offset, int64_t index_rows,
                             int64_t index_columns, int64_t index_rs,
                             int64_t index_cs) {
  const int32_t expected_values[3] = {16777216, 8388608, 8388608};
  const int64_t expected_indices[3] = {2, 1, 4};
  if (rows != 1 || columns != 3 || index_rows != 1 || index_columns != 3)
    exit(1);
  for (int i = 0; i < 3; ++i)
    if (values[offset + i * cs] != expected_values[i] ||
        indices[index_offset + i * index_cs] != expected_indices[i]) {
      printf("topk_i32_scores FAIL rank=%d index=%ld\n", i,
             indices[index_offset + i * index_cs]);
      exit(1);
    }
  printf("topk_i32_scores PASS score bits and stable I64 indices\n");
}
