// A tile's share of a general matrix product of the mesh: Z = Z + X * W in
// FP16, X of M x N, W of N x K, Z of M x K, in L2. The rows of X are
// contiguous; W and Z are K columns of matrices whose rows are `pitch`
// columns apart, so that a product may take a block of columns of wider
// matrices. The tiles share the columns of W and Z: a tile copies X and its
// columns to its L1, runs RedMulE and copies its columns of Z back. A tile
// takes at least two columns (RedMulE does nothing with one), so a product
// of few columns keeps some tiles idle; it has at least two.
#ifndef MAGIA_FIRMWARE_GEMM_H
#define MAGIA_FIRMWARE_GEMM_H

#include "idma_mm_utils.h"
#include "magia_tile_utils.h"
#include "redmule_mm_utils.h"
#include <stdint.h>

struct magia_job {
  uint32_t x, w, z;
  uint32_t m, n, k;
  uint32_t pitch;
};

// The job of a launch, in L2 between the firmware's instructions and data.
#define MAGIA_JOB 0xcc00f000

// The matrices of a job in a tile's L1, after the first words the tile's
// runtime keeps.
#define MAGIA_GEMM_L1 0x12048

static inline uint32_t magia_words(uint32_t bytes) { return (bytes + 3) & ~3u; }

// Returns nonzero when the job does not fit the tile or the hardware stalls.
static inline int magia_tile_gemm(const struct magia_job *job, uint32_t tile,
                                  uint32_t tiles) {
  uint32_t busy = job->k / 2 < tiles ? job->k / 2 : tiles;
  if (tile >= busy)
    return 0;
  uint32_t first = job->k * tile / busy;
  uint32_t columns = job->k * (tile + 1) / busy - first;
  uint32_t l1 = L1_BASE + tile * L1_TILE_OFFSET;
  uint32_t x = l1 + MAGIA_GEMM_L1;
  uint32_t w = x + magia_words(job->m * job->n * 2);
  uint32_t z = w + magia_words(job->n * columns * 2);
  if (z + job->m * columns * 2 > l1 + L1_SIZE)
    return 1;
  // Rows of X whole; rows of W and Z by this tile's columns.
  if (!idma_mm_wait_for_completion(
          IDMA_DIR_L2_TO_L1, idma_L2ToL1_2d(job->x, x, job->n * 2, job->n * 2,
                                            job->n * 2, job->m)) ||
      !idma_mm_wait_for_completion(IDMA_DIR_L2_TO_L1,
                                   idma_L2ToL1_2d(job->w + first * 2, w,
                                                  columns * 2, job->pitch * 2,
                                                  columns * 2, job->n)) ||
      !idma_mm_wait_for_completion(IDMA_DIR_L2_TO_L1,
                                   idma_L2ToL1_2d(job->z + first * 2, z,
                                                  columns * 2, job->pitch * 2,
                                                  columns * 2, job->m)))
    return 1;
  hwpe_soft_clear();
  while (hwpe_acquire_job() < 0) {
  }
  redmule_cfg(x, w, z, job->m, job->n, columns, (uint8_t)gemm_ops,
              (uint8_t)Float16, (uint8_t)Float16);
  hwpe_trigger_job();
  if (hwpe_wait_for_completion())
    return 1;
  return !idma_mm_wait_for_completion(
      IDMA_DIR_L1_TO_L2, idma_L1ToL2_2d(z, job->z + first * 2, columns * 2,
                                        columns * 2, job->pitch * 2, job->m));
}

#endif // MAGIA_FIRMWARE_GEMM_H
