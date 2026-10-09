#include <bbhw/isa/isa.h>
#include <params.h>
#include <topology.h>

#include <stdint.h>

#if BANK_WIDTH != 128
#error "mesh_shared_mem_test requires 128-bit Bank rows"
#endif
#if BB_SHARED_PHYSICAL_BANK_NUM < 1
#error "mesh_shared_mem_test requires at least one shared Bank"
#endif

#if BB_VIRTUAL_BANK_NUM <= BB_SHARED_BANK_BASE
#error "mesh_shared_mem_test requires shared virtual Bank IDs"
#endif
#if BB_VIRTUAL_BANK_NUM - BB_SHARED_BANK_BASE > BB_SHARED_PHYSICAL_BANK_NUM
#error "mesh_shared_mem_test requires a physical Bank per shared virtual Bank"
#endif

enum {
  SHARED_TEST_BANKS = (BB_VIRTUAL_BANK_NUM - BB_SHARED_BANK_BASE),
  ROW_BYTES = (BANK_WIDTH / 8)
};

static uint8_t source[SHARED_TEST_BANKS][ROW_BYTES]
    __attribute__((aligned(64)));
static uint8_t result[SHARED_TEST_BANKS][ROW_BYTES]
    __attribute__((aligned(64)));

/* crt0 boots only the target's test hart. */
int main(void) {
  for (int bank = 0; bank < SHARED_TEST_BANKS; ++bank) {
    bb_mem_alloc(BB_SHARED_BANK_BASE + bank, 1, 1);
    for (int byte = 0; byte < ROW_BYTES; ++byte)
      source[bank][byte] = (uint8_t)(bank * 17 + byte);
    bb_mvin((uintptr_t)source[bank], BB_SHARED_BANK_BASE + bank, 1, 1);
  }

  for (int bank = 0; bank < SHARED_TEST_BANKS; ++bank)
    bb_mvout((uintptr_t)result[bank], BB_SHARED_BANK_BASE + bank, 1, 1);

  for (int bank = 0; bank < SHARED_TEST_BANKS; ++bank) {
    for (int byte = 0; byte < ROW_BYTES; ++byte)
      if (result[bank][byte] != source[bank][byte])
        return 1;
    bb_mem_release(BB_SHARED_BANK_BASE + bank);
  }
  return 0;
}
