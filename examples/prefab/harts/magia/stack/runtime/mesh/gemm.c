// magia_gemm on bare metal: the program runs at the DDR's addresses, so the
// matrices and L2 are where the mesh has them.
#include "jobs.h"

void magia_gemm(uint64_t x, uint64_t w, uint64_t z, uint64_t batch, uint64_t m,
                uint64_t n, uint64_t k) {
  magia_jobs((volatile uint8_t *)MAGIA_L2, x, w, z, batch, m, n, k);
}
