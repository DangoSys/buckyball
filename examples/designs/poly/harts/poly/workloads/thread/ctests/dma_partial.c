#include <bbhw/isa/isa.h>
#include <stdint.h>

uint64_t entry(const uint64_t *addresses) {
  bb_mem_alloc(1, 1, 1);
  bb_mvin(addresses[0], 1, 4, 1);
  bb_mvout_strided(addresses[1], 1, 0, 0, 1, 2, 20, 16, 0);
  bb_mvout_strided(addresses[1] + 16, 1, 0, 2, 1, 2, 20, 4, 0);
  bb_mem_release(1);
  bb_mem_alloc(1, 1, 1);
  bb_mvin_strided(addresses[0] + 3, 1, 0, 0, 1, 8, 16, 16, 0);
  bb_mvin_strided(addresses[0] + 131, 1, 0, 8, 1, 1, 16, 4, 0);
  bb_mvout_strided(addresses[1] + 48, 1, 0, 8, 1, 1, 16, 16, 0);
  bb_mem_release(1);
  return 0;
}
