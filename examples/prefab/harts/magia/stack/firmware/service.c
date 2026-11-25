// The firmware of the mesh for the host's matrix products: a launch is one
// job, which the host left in L2 (runtime/mesh/gemm.c); every tile does its
// share and returns whether it failed.
#include "gemm.h"
#include "magia_utils.h"

int main(void) {
  return magia_tile_gemm((const struct magia_job *)MAGIA_JOB, get_hartid(),
                         NUM_HARTS);
}
