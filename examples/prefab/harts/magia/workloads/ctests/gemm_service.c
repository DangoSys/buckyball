// The host's matrix product on the mesh, through the runtime
// (stack/runtime/mesh/gemm.c) and the firmware (stack/firmware/service.c):
// 4 x 16 x 64 in FP16, four columns a tile. The matrices lie below L2: the
// mesh reaches them through its windows.
//
// The values are small integers, exact in FP16, with a product the host has
// in closed form (it takes about 13 cycles an instruction): X is 1 but for a
// 2 in column m of row m, W[n][k] is (n + k) mod 4, whose columns each sum to
// 6 N / 4, and Z starts at (5 m + 3 k) mod 8. So
// Z[m][k] = (5 m + 3 k) mod 8 + 6 N / 4 + (m + k) mod 4.
#include <stdint.h>
#include <stdio.h>

#define M_SIZE 4
#define N_SIZE 16
#define K_SIZE 64

void magia_gemm(uint64_t x, uint64_t w, uint64_t z, uint64_t m, uint64_t n,
                uint64_t k);

static uint16_t x[M_SIZE * N_SIZE];
static uint16_t w[N_SIZE * K_SIZE];
static uint16_t z[M_SIZE * K_SIZE];

// The FP16 bits of an integer from 0 to 63.
static uint16_t half(unsigned value) {
  if (value == 0)
    return 0;
  unsigned top = 5;
  while (!(value >> top))
    --top;
  return (uint16_t)((top + 15) << 10) |
         (uint16_t)((value << (10 - top)) & 0x3ff);
}

// Every value of the test, encoded once.
static uint16_t halves[8 + 6 * N_SIZE / 4 + 4];

int main(void) {
  for (unsigned value = 0; value < sizeof(halves) / sizeof(halves[0]); ++value)
    halves[value] = half(value);
  for (unsigned m = 0; m < M_SIZE; ++m)
    for (unsigned n = 0; n < N_SIZE; ++n)
      x[m * N_SIZE + n] = halves[n == m ? 2 : 1];
  for (unsigned n = 0; n < N_SIZE; ++n)
    for (unsigned k = 0; k < K_SIZE; ++k)
      w[n * K_SIZE + k] = halves[(n + k) & 3];
  for (unsigned m = 0; m < M_SIZE; ++m)
    for (unsigned k = 0; k < K_SIZE; ++k)
      z[m * K_SIZE + k] = halves[(5 * m + 3 * k) & 7];
  uint64_t start, end;
  __asm__ volatile("rdcycle %0" : "=r"(start));
  magia_gemm((uint64_t)x, (uint64_t)w, (uint64_t)z, M_SIZE, N_SIZE, K_SIZE);
  __asm__ volatile("rdcycle %0" : "=r"(end));
  // The rows that are wrong in each column, a hexadecimal digit a column.
  char wrong[K_SIZE + 1];
  unsigned errors = 0;
  for (unsigned k = 0; k < K_SIZE; ++k) {
    unsigned rows = 0;
    for (unsigned m = 0; m < M_SIZE; ++m)
      if (z[m * K_SIZE + k] !=
          halves[((5 * m + 3 * k) & 7) + 6 * N_SIZE / 4 + ((m + k) & 3)]) {
        if (errors < 4)
          printf(
              "Mesh product: row %u column %u got %04x expected %04x\n", m, k,
              z[m * K_SIZE + k],
              halves[((5 * m + 3 * k) & 7) + 6 * N_SIZE / 4 + ((m + k) & 3)]);
        rows |= 1u << m;
        ++errors;
      }
    wrong[k] = "0123456789abcdef"[rows];
  }
  wrong[K_SIZE] = 0;
  if (errors)
    printf("Mesh product: wrong rows by column %s\n", wrong);
  printf("Mesh product: 4x16x64 through the runtime, %u errors, %lu cycles\n",
         errors, (unsigned long)(end - start));
  return errors != 0;
}
