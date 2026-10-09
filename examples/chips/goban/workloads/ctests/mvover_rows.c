#include <bbhw/isa/isa.h>
#include <multicore.h>
#include <params.h>
#include <stdint.h>
#include <topology.h>

#define ROWS 8
#define ROW_BYTES (BANK_WIDTH / 8)
#if BANK_LINES < ROWS
#error "mvover_rows requires eight rows per bank"
#endif
static uint8_t input[ROWS * ROW_BYTES] __attribute__((aligned(64)));
static uint8_t output[ROWS * ROW_BYTES] __attribute__((aligned(64)));
#if BB_SHARED_PHYSICAL_BANK_NUM < 1 || BB_SHARED_BANK_BASE >= VIRTUAL_BANK_NUM
#error "mvover_rows requires a shared bank for issuer/owner trace coverage"
#endif
static uint8_t shared_output[ROWS * ROW_BYTES] __attribute__((aligned(64)));
static unsigned source_ready, target_ready, moved, stored;
int main(void) {
  core_id_t id = bb_get_core_id();
  if (id.tile != 0)
    for (;;)
      asm volatile("wfi");
  if (id.core == 1) {
    for (unsigned i = 0; i < sizeof(input); ++i)
      input[i] = (uint8_t)(i * 17 + 3);
    bb_mem_alloc(1, 1, 1);
    bb_mem_alloc(BB_SHARED_BANK_BASE, 1, 1);
    bb_mvin((uintptr_t)input, 1, ROWS, 1);
    // Hart1 issues the write; the shared storage belongs to this Tile's Core0.
    bb_mvin((uintptr_t)input, BB_SHARED_BANK_BASE, ROWS, 1);
    bb_mvout((uintptr_t)shared_output, BB_SHARED_BANK_BASE, ROWS, 1);
    for (unsigned i = 0; i < sizeof(input); ++i)
      if (shared_output[i] != input[i])
        return 1;
    bb_mem_release(BB_SHARED_BANK_BASE);
    __atomic_store_n(&source_ready, 1, __ATOMIC_RELEASE);
  } else if (id.core == 2) {
    bb_mem_alloc(8, 1, 1);
    __atomic_store_n(&target_ready, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&moved, __ATOMIC_ACQUIRE)) {
    }
    bb_mvout((uintptr_t)output, 8, ROWS, 1);
    __atomic_store_n(&stored, 1, __ATOMIC_RELEASE);
  } else if (id.core == 0) {
    while (!__atomic_load_n(&source_ready, __ATOMIC_ACQUIRE) ||
           !__atomic_load_n(&target_ready, __ATOMIC_ACQUIRE)) {
    }
    bb_mvover(0, 1, 0, 1, 8, 0, ROWS);
    __atomic_store_n(&moved, 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n(&stored, __ATOMIC_ACQUIRE)) {
    }
    for (unsigned i = 0; i < sizeof(input); ++i)
      if (input[i] != output[i])
        return 1;
    return 0;
  }
  for (;;)
    asm volatile("wfi");
}
