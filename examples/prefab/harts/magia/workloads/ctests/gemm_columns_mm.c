// A general matrix product shared by the tiles: the first 4 rows and 32
// columns of RedMulE's 16 x 64 x 64 golden product in FP16, every tile the two
// columns of W and Z that are its share (stack/firmware/gemm.h), X, W and Z in
// L2. W and Z stay 64 columns wide: the product leaves the other columns of Z
// alone. The host starts the mesh twice: tile 0 sets Z to Y at the start of
// each run. The data of the test is small: a tile takes about a hundred cycles
// to reach a word of L2, and tile 0 copies and checks every word of Z alone.
#include "fsync_mm_utils.h"
#include "gemm.h"
#include "magia_tile_utils.h"
#include "magia_utils.h"
#include "w_input.h"
#include "x_input.h"
#include "y_input.h"
#include "z_output.h"
#include <stdint.h>

#define M_SIZE 4
#define N_SIZE 64
#define K_SIZE 32
#define PITCH 64
#define DIFF_TH 0x0011

static volatile uint32_t failed[NUM_HARTS];
static uint16_t z_job[M_SIZE * PITCH];

int main(void) {
  uint32_t tile = get_hartid();
  if (tile == 0) {
    for (unsigned i = 0; i < M_SIZE * PITCH; ++i)
      z_job[i] = y_inp[i];
    asm volatile("fence rw, rw" : : : "memory");
  }
  fsync_mm(0, NUM_HARTS - 1);
  struct magia_job job = {(uint32_t)x_inp,
                          (uint32_t)w_inp,
                          (uint32_t)z_job,
                          M_SIZE,
                          N_SIZE,
                          K_SIZE,
                          PITCH};
  failed[tile] = magia_tile_gemm(&job, tile, NUM_HARTS);
  asm volatile("fence rw, rw" : : : "memory");
  fsync_mm(0, NUM_HARTS - 1);
  if (tile != 0)
    return failed[tile];

  unsigned errors = 0;
  for (unsigned other = 0; other < NUM_HARTS; ++other)
    errors += failed[other];
  for (unsigned row = 0; row < M_SIZE; ++row)
    for (unsigned column = 0; column < PITCH; ++column) {
      uint16_t computed = z_job[row * PITCH + column];
      // Outside the product Z keeps Y, bit for bit.
      uint16_t expected = column < K_SIZE ? z_oup[row * PITCH + column]
                                          : y_inp[row * PITCH + column];
      unsigned diff =
          computed > expected ? computed - expected : expected - computed;
      if (diff > (column < K_SIZE ? DIFF_TH : 0)) {
        if (errors == 0)
          printf("Mesh GEMM: row %u column %u got %04x expected %04x\n", row,
                 column, computed, expected);
        ++errors;
      }
    }
  printf("Mesh GEMM: 4x64x32 over %u tiles, %u errors\n", NUM_HARTS, errors);
  return errors;
}
