#include <cmath>
#include <cstdint>
#include <kernel.h>

struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_compute_decode(matrix *, matrix *);
alignas(64) static int8_t decode_input[32 * 137 + 8];
alignas(64) static float decode_output[32 * 132 + 8];
uint64_t decode_cycles[2];

int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  const int decode_heights[] = {1, 5, 32};
  for (int rows : decode_heights) {
    for (int i = 0; i < 32 * 137 + 8; ++i)
      decode_input[i] = 0x35;
    for (float &x : decode_output)
      x = 17;
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < 128; ++col)
        decode_input[3 + row * 137 + col] = (row * 29 + col * 7) % 255 - 127;
      for (int block = 0; block < 4; ++block)
        decode_input[3 + row * 137 + 128 + block] =
            ((row * 4 + block) * 13) % 255;
    }
    matrix a{decode_input, decode_input, 3, {rows, 132}, {137, 1}};
    matrix out{decode_output, decode_output, 4, {rows, 128}, {132, 1}};
    asm volatile("rdcycle %0" : "=r"(decode_cycles[0]));
    _mlir_ciface_compute_decode(&a, &out);
    asm volatile("rdcycle %0" : "=r"(decode_cycles[1]));
    for (int i = 0; i < 32 * 132 + 8; ++i) {
      int row = (i - 4) / 132, col = (i - 4) % 132;
      float expected = 17;
      if (i >= 4 && row < rows && col < 128)
        expected = -ldexpf(
            decode_input[3 + row * 137 + col],
            uint8_t(decode_input[3 + row * 137 + 128 + col / 32]) - 127);
      if (decode_output[i] != expected ||
          std::signbit(decode_output[i]) != std::signbit(expected))
        return 6;
    }
  }
  return 0;
}
