#include "ant_bank.h"
#include <ant.h>
#include <params.h>
#include <topology.h>
extern const unsigned char ant_image[], ant_image_end[];
enum {
  WORDS = BANK_LINES * BANK_WIDTH / 64,
  SHARED_WORDS = BB_SHARED_BANK_LINES * BANK_WIDTH / 64
};
static uint64_t input[BB_COMPUTE_TILES][WORDS]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t private_out[BB_COMPUTE_TILES][WORDS]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t shared_out[BB_COMPUTE_TILES][SHARED_WORDS]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t tail_out[BB_COMPUTE_TILES][8]
    __attribute__((section(".data"), aligned(4096))) = {0};
unsigned bank_done[BB_COMPUTE_TILES];
void bank_exercise(unsigned tile) {
  const unsigned context = BB_ANT_CONTEXT;
  for (unsigned i = 0; i < WORDS; ++i)
    input[tile][i] = (uint64_t)(tile + 1) << 32 | (i + 1);
  for (unsigned i = 0; i < WORDS; ++i)
    private_out[tile][i] = 0;
  for (unsigned i = 0; i < SHARED_WORDS; ++i)
    shared_out[tile][i] = 0;
  for (unsigned i = 0; i < 8; ++i)
    tail_out[tile][i] = ~0ULL;
  size_t bytes = ant_image_end - ant_image;
  if (bytes > ant_query(context, ANT_CODE_BYTES) ||
      ant_query(context, ANT_SIGNATURE) != CORE_SIGNATURE)
    __builtin_trap();
  struct ant_bank_args args = {(uintptr_t)input[tile],
                               (uintptr_t)private_out[tile],
                               (uintptr_t)shared_out[tile],
                               (uintptr_t)tail_out[tile], BB_SHARED_BANK_LINES};
  ant_write(context, ANT_CODE, 0, ant_image, bytes);
  ant_write(context, ANT_TLS, 0, &args, sizeof(args));
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  struct ant_task task = {
      tile + 1,      0, bytes, base, base + ant_query(context, ANT_TLS_BYTES),
      CORE_SIGNATURE};
  ant_acquire();
  ant_start(context, &task);
  int cancelled;
  if (ant_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  ant_release();
  for (unsigned i = 0; i < WORDS; ++i)
    if (private_out[tile][i] != (((uint64_t)(tile + 1) << 32) | (i + 1)))
      __builtin_trap();
  for (unsigned i = 0; i < SHARED_WORDS; ++i)
    if (shared_out[tile][i] != (((uint64_t)(tile + 1) << 32) | (i + 1)))
      __builtin_trap();
  for (unsigned i = 0; i < 8; ++i)
    if (tail_out[tile][i] !=
        (i % 2 ? 0 : (((uint64_t)(tile + 1) << 32) | (i + 1))))
      __builtin_trap();
  __atomic_store_n(&bank_done[tile], 1, __ATOMIC_RELEASE);
}
