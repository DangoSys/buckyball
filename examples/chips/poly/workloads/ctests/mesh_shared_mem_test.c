#include <bbhw/isa/isa.h>
#include <multicore.h>
#include <params.h>
#include <topology.h>

#include <stdint.h>

#if BANK_WIDTH != 128
#error "mesh_shared_mem_test requires 128-bit Bank rows"
#endif
#if BB_SHARED_PHYSICAL_BANK_NUM < 1
#error "mesh_shared_mem_test requires at least one shared Bank"
#endif

#define ROW_BYTES (BANK_WIDTH / 8)

static uint8_t source[BB_SHARED_PHYSICAL_BANK_NUM][ROW_BYTES]
    __attribute__((aligned(64)));
static uint8_t result[BB_SHARED_PHYSICAL_BANK_NUM][ROW_BYTES]
    __attribute__((aligned(64)));

int main(void) {
  core_id_t id = bb_get_core_id();
  if (id.tile != 0 || id.core != 0)
    for (;;)
      asm volatile("wfi");

  for (int bank = 0; bank < BB_SHARED_PHYSICAL_BANK_NUM; ++bank) {
    bb_mem_alloc(BB_SHARED_BANK_BASE + bank, 1, 1);
    for (int byte = 0; byte < ROW_BYTES; ++byte)
      source[bank][byte] = (uint8_t)(bank * 17 + byte);
    bb_mvin((uintptr_t)source[bank], BB_SHARED_BANK_BASE + bank, 1, 1);
  }
  bb_fence();

  for (int bank = 0; bank < BB_SHARED_PHYSICAL_BANK_NUM; ++bank)
    bb_mvout((uintptr_t)result[bank], BB_SHARED_BANK_BASE + bank, 1, 1);
  bb_fence();

  for (int bank = 0; bank < BB_SHARED_PHYSICAL_BANK_NUM; ++bank) {
    for (int byte = 0; byte < ROW_BYTES; ++byte)
      if (result[bank][byte] != source[bank][byte])
        return 1;
    bb_mem_release(BB_SHARED_BANK_BASE + bank);
  }
  bb_fence();
  return 0;
}
