// magia_gemm under Linux. The mesh reaches the DDR by its addresses, a
// contiguous range a window, and a program's memory is neither. So the image
// of the device keeps a part of the DDR from Linux (its device tree): L2's
// firmware and, behind it, a pool; the program maps both through /dev/mem.
// A matrix in memory the program registered (magia_region: the weights of a
// model, mapped from the DDR) is used where it is. Any other is copied into
// the pool for the product, and Z back: two bytes an element, next to the
// conversion to FP16 that made it.
#include "jobs.h"
#include <fcntl.h>
#include <string.h>
#include <sys/mman.h>

// The reserved part of the DDR: 16 MiB of L2 for the firmware, then the pool.
#define MAGIA_POOL UINT64_C(0x1000000)
#define MAGIA_RESERVED UINT64_C(0x5000000)

// The firmware of the tiles (firmware/linux.S).
extern const unsigned char magia_firmware[], magia_firmware_end[];

struct magia_region {
  uint64_t address, physical, bytes;
};
static struct magia_region regions[4];
static unsigned region_count;
static uint8_t *l2;

// Memory of the program that is `bytes` of the DDR at `physical`.
void magia_region(const void *address, uint64_t physical, uint64_t bytes) {
  if (region_count == sizeof(regions) / sizeof(regions[0]))
    abort();
  regions[region_count++] =
      (struct magia_region){(uint64_t)address, physical, bytes};
}

// The DDR address of a matrix: its own in a region, else of a copy in the
// pool after the `*used` bytes other matrices of the product took.
static uint64_t place(uint64_t address, uint64_t bytes, uint64_t *used) {
  for (unsigned index = 0; index < region_count; ++index)
    if (address >= regions[index].address &&
        address + bytes <= regions[index].address + regions[index].bytes)
      return regions[index].physical + address - regions[index].address;
  uint64_t offset = MAGIA_POOL + *used;
  if (offset + bytes > MAGIA_RESERVED)
    abort();
  *used += (bytes + 0xfff) & ~UINT64_C(0xfff);
  memcpy(l2 + offset, (const void *)address, bytes);
  return MAGIA_L2 + offset;
}

void magia_gemm(uint64_t x, uint64_t w, uint64_t z, uint64_t batch, uint64_t m,
                uint64_t n, uint64_t k) {
  if (!l2) {
    int memory = open("/dev/mem", O_RDWR);
    if (memory < 0)
      abort();
    l2 = mmap(NULL, MAGIA_RESERVED, PROT_READ | PROT_WRITE, MAP_SHARED, memory,
              MAGIA_L2);
    if (l2 == MAP_FAILED)
      abort();
    memcpy(l2, magia_firmware, magia_firmware_end - magia_firmware);
  }
  uint64_t used = 0;
  uint64_t xs = place(x, batch * m * n * 2, &used);
  uint64_t ws = place(w, batch * n * k * 2, &used);
  uint64_t zs = place(z, batch * m * k * 2, &used);
  magia_jobs(l2, xs, ws, zs, batch, m, n, k);
  if (zs - MAGIA_L2 < MAGIA_RESERVED)
    memcpy((void *)z, l2 + (zs - MAGIA_L2), batch * m * k * 2);
}
