#include "ant_bank.h"
#include <thread.h>
extern const unsigned char ant_image[], ant_image_end[];
enum { WORDS = THREAD_MEM_BANK_ENTRIES * THREAD_MEM_BANK_WIDTH / 64 };
static uint64_t input[POLY_THREAD_HARTS][WORDS]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t private_out[POLY_THREAD_HARTS][WORDS]
    __attribute__((section(".data"), aligned(4096))) = {0};
static uint64_t tail_out[POLY_THREAD_HARTS][8]
    __attribute__((section(".data"), aligned(4096))) = {0};
unsigned bank_done[POLY_THREAD_HARTS];
void bank_exercise(unsigned hart) {
  const unsigned context = thread_find(THREAD_SIGNATURE, 0);
  for (unsigned i = 0; i < WORDS; ++i)
    input[hart][i] = (uint64_t)(hart + 1) << 32 | (i + 1);
  for (unsigned i = 0; i < WORDS; ++i)
    private_out[hart][i] = 0;
  for (unsigned i = 0; i < 8; ++i)
    tail_out[hart][i] = ~0ULL;
  size_t bytes = ant_image_end - ant_image;
  if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
      thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
    __builtin_trap();
  struct ant_bank_args args = {(uintptr_t)input[hart],
                               (uintptr_t)private_out[hart],
                               (uintptr_t)tail_out[hart]};
  thread_tss_write(0, ant_image, bytes);
  thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  thread_tss_write(0, &args, sizeof(args));
  thread_load(context, THREAD_TLS, 0, 0, (sizeof(args) + 7) & ~7u);
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  struct thread_task task = {hart + 1,
                             0,
                             bytes,
                             base,
                             base +
                                 thread_query(context, THREAD_FIELD_TLS_BYTES),
                             THREAD_SIGNATURE};
  thread_acquire();
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, task.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
  for (unsigned i = 0; i < WORDS; ++i)
    if (private_out[hart][i] != (((uint64_t)(hart + 1) << 32) | (i + 1)))
      __builtin_trap();
  for (unsigned i = 0; i < 8; ++i)
    if (tail_out[hart][i] !=
        (i % 2 ? 0 : (((uint64_t)(hart + 1) << 32) | (i + 1))))
      __builtin_trap();
  __atomic_store_n(&bank_done[hart], 1, __ATOMIC_RELEASE);
}
