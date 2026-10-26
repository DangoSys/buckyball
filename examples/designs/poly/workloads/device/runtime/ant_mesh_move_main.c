#define _GNU_SOURCE
#include "ant_mesh_move.h"
#include <instances.h>
#include <params.h>
#include <thread.h>

extern const unsigned char ant_image[], ant_image_end[];
enum { ROW_WORDS = BANK_WIDTH / 64, ROWS = 2 };
static uint64_t input[BB_THREAD_CORE_COUNT][ROWS * ROW_WORDS]
    __attribute__((section(".data"), aligned(4096)));
static uint64_t output[BB_THREAD_CORE_COUNT][ROWS * ROW_WORDS]
    __attribute__((section(".data"), aligned(4096)));
static unsigned done[BB_THREAD_CORE_COUNT];

static void run(unsigned context, unsigned id,
                struct ant_mesh_move_args *args) {
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  thread_tss_write(64, args, sizeof(*args));
  thread_load(context, THREAD_TLS, 0, 64, sizeof(*args));
  struct thread_task task = {id,
                             0,
                             ant_image_end - ant_image,
                             base,
                             base +
                                 thread_query(context, THREAD_FIELD_TLS_BYTES),
                             THREAD_SIGNATURE};
  thread_start(context, &task);
  int cancelled;
  if (thread_wait(context, id, &cancelled) || cancelled)
    __builtin_trap();
}

static struct thread_task prepare(unsigned context, unsigned id,
                                  struct ant_mesh_move_args *args) {
  uint64_t base = thread_query(context, THREAD_FIELD_TLS_BASE);
  thread_tss_write(64, args, sizeof(*args));
  thread_load(context, THREAD_TLS, 0, 64, sizeof(*args));
  return (struct thread_task){id,
                              0,
                              ant_image_end - ant_image,
                              base,
                              base +
                                  thread_query(context, THREAD_FIELD_TLS_BYTES),
                              THREAD_SIGNATURE};
}

static void exercise(unsigned tile) {
  size_t bytes = ant_image_end - ant_image;
  for (unsigned context = 0; context < 2; ++context) {
    if (bytes > thread_query(context, THREAD_FIELD_CODE_BYTES) ||
        thread_query(context, THREAD_FIELD_SIGNATURE) != THREAD_SIGNATURE)
      __builtin_trap();
    thread_tss_write(0, ant_image, bytes);
    thread_load(context, THREAD_CODE, 0, 0, (bytes + 7) & ~7u);
  }
  for (unsigned i = 0; i < ROWS * ROW_WORDS; ++i) {
    input[tile][i] = ((uint64_t)(tile + 1) << 56) | (0x100 + i);
    output[tile][i] = 0;
  }
  uint64_t zero[2] = {0};
  thread_tss_write(0, zero, sizeof(zero));
  struct ant_mesh_move_args args = {
      (uintptr_t)input[tile], (uintptr_t)output[tile],
      thread_query(0, THREAD_FIELD_TSS_BASE), MESH_FILL};
  thread_acquire();
  run(0, tile * 3 + 1, &args);
  args.role = MESH_MOVE;
  struct thread_task move = prepare(0, tile * 3 + 2, &args);
  args.role = MESH_CONSUME;
  struct thread_task consume = prepare(1, tile * 3 + 3, &args);
  thread_start(1, &consume);
  thread_start(0, &move);
  int cancelled;
  if (thread_wait(0, move.id, &cancelled) || cancelled ||
      thread_wait(1, consume.id, &cancelled) || cancelled)
    __builtin_trap();
  thread_release();
  for (unsigned i = 0; i < ROWS * ROW_WORDS; ++i)
    if (output[tile][i] != input[tile][i])
      __builtin_trap();
  __atomic_store_n(&done[tile], 1, __ATOMIC_RELEASE);
}

int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != BB_REPORT_HART)
    exercise(hart - 1);
  else
    for (unsigned tile = 0; tile < BB_THREAD_CORE_COUNT; ++tile)
      while (!__atomic_load_n(&done[tile], __ATOMIC_ACQUIRE)) {
      }
  return 0;
}
