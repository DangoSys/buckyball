// A matrix product as jobs of the mesh: the host leaves a job in L2, starts
// the mesh with the firmware (firmware/service.c, at 0xcc000000) and waits
// for the tiles. The matrices lie anywhere in the DDR, each contiguous: the
// mesh sees them through three windows of L2 that Loom points at them. A
// product of more rows or columns than the tiles hold goes in several jobs.
#include <stdint.h>
#include <stdlib.h>

#define MAGIA_L2 UINT64_C(0xcc000000)
// In L2: the entry of the firmware, its job, and the first word of its data,
// which tile 0 sets once it has cleared the data of a launch (the other tiles
// wait for it).
#define MAGIA_ENTRY UINT64_C(0x80)
#define MAGIA_JOB UINT64_C(0xf000)
#define MAGIA_STARTED UINT64_C(0x10000)
#define MAGIA_TILES 16
// A tile's L1 after the first words its runtime keeps (firmware/gemm.h).
#define MAGIA_L1_BYTES (0xe0000 - 0x12048)
// The windows of the mesh onto the DDR: W, X and Z.
#define MAGIA_W UINT64_C(0xd0000000)
#define MAGIA_W_BYTES UINT64_C(0x20000000)
#define MAGIA_X UINT64_C(0xf0000000)
#define MAGIA_Z UINT64_C(0xf8000000)
#define MAGIA_XZ_BYTES UINT64_C(0x08000000)

struct magia_job {
  uint32_t x, w, z;
  uint32_t m, n, k;
  uint32_t pitch;
};

// Points window `index` at the page of the DDR address `address` and returns
// where the mesh sees `address`; the matrix of `bytes` must fit the window.
static uint32_t magia_window(uint64_t index, uint64_t base, uint64_t limit,
                             uint64_t address, uint64_t bytes) {
  uint64_t page = address & ~UINT64_C(0xfff);
  if (address - page + bytes > limit)
    abort();
  __asm__ volatile(".insn r 0x0b, 3, 3, x0, %0, %1"
                   :
                   : "r"(page), "r"(index)
                   : "memory");
  return (uint32_t)(base + address - page);
}

// `l2` is where the program sees L2.
static void magia_launch(volatile uint8_t *l2) {
  // The flag of the launch before would let the tiles run ahead of tile 0.
  *(volatile uint32_t *)(l2 + MAGIA_STARTED) = 0;
  __asm__ volatile("fence rw, rw" : : : "memory");
  uint64_t entry = MAGIA_L2 + MAGIA_ENTRY;
  __asm__ volatile(".insn r 0x0b, 2, 0, x0, %0, x0" : : "r"(entry) : "memory");
  uint64_t status;
  do
    __asm__ volatile(".insn r 0x0b, 4, 1, %0, x0, x0"
                     : "=r"(status)
                     :
                     : "memory");
  while (!(status & 1));
  // A tile that failed stops the program: the result would be wrong.
  if (status >> 32)
    abort();
}

// Z = Z + X * W with the matrices at the DDR addresses `x`, `w` and `z`.
static void magia_jobs(volatile uint8_t *l2, uint64_t x, uint64_t w, uint64_t z,
                       uint64_t m, uint64_t n, uint64_t k) {
  // A tile takes at least two columns (firmware/gemm.h).
  if (k < 2)
    abort();
  // The columns of a tile: its share of W takes at most half of its L1, its
  // rows of X and Z the rest. More columns than that go in several jobs of
  // about the same width.
  uint64_t busy = k / 2 < MAGIA_TILES ? k / 2 : MAGIA_TILES;
  uint64_t columns = (k + busy - 1) / busy;
  uint64_t jobs = 1;
  if (columns > MAGIA_L1_BYTES / 2 / (n * 2)) {
    columns = MAGIA_L1_BYTES / 2 / (n * 2);
    if (!columns)
      abort();
    jobs = (k + columns * MAGIA_TILES - 1) / (columns * MAGIA_TILES);
  }
  uint64_t rows =
      (MAGIA_L1_BYTES - n * columns * 2 - 8) / (n * 2 + columns * 2);
  if (!rows)
    abort();
  volatile struct magia_job *job =
      (volatile struct magia_job *)(l2 + MAGIA_JOB);
  uint32_t ws = magia_window(0, MAGIA_W, MAGIA_W_BYTES, w, n * k * 2);
  for (uint64_t top = 0; top < m; top += rows) {
    uint64_t height = m - top < rows ? m - top : rows;
    uint32_t xs = magia_window(1, MAGIA_X, MAGIA_XZ_BYTES, x + top * n * 2,
                               height * n * 2);
    uint32_t zs = magia_window(2, MAGIA_Z, MAGIA_XZ_BYTES, z + top * k * 2,
                               height * k * 2);
    for (uint64_t index = 0; index < jobs; ++index) {
      uint64_t left = k * index / jobs;
      job->x = xs;
      job->w = ws + (uint32_t)(left * 2);
      job->z = zs + (uint32_t)(left * 2);
      job->m = (uint32_t)height;
      job->n = (uint32_t)n;
      job->k = (uint32_t)(k * (index + 1) / jobs - left);
      job->pitch = (uint32_t)k;
      magia_launch(l2);
    }
  }
}
