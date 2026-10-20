#define _GNU_SOURCE
#include "ant_mesh_move.h"
#include <ant.h>
#include <params.h>
#include <topology.h>

extern const unsigned char ant_image[], ant_image_end[];
enum { ROW_WORDS = BANK_WIDTH / 64, ROWS = 2 };
static uint64_t input[BB_COMPUTE_TILES][ROWS * ROW_WORDS]
    __attribute__((section(".data"), aligned(4096)));
static uint64_t output[BB_COMPUTE_TILES][ROWS * ROW_WORDS]
    __attribute__((section(".data"), aligned(4096)));
static unsigned done[BB_COMPUTE_TILES];

static void run(unsigned context, unsigned id,
                struct ant_mesh_move_args *args) {
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  ant_write(context, ANT_TLS, 0, args, sizeof(*args));
  struct ant_task task = {id,
                          0,
                          ant_image_end - ant_image,
                          base,
                          base + ant_query(context, ANT_TLS_BYTES),
                          CORE_SIGNATURE};
  ant_start(context, &task);
  int cancelled;
  if (ant_wait(context, id, &cancelled) || cancelled)
    __builtin_trap();
}

static struct ant_task prepare(unsigned context, unsigned id,
                               struct ant_mesh_move_args *args) {
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  ant_write(context, ANT_TLS, 0, args, sizeof(*args));
  return (struct ant_task){id,
                           0,
                           ant_image_end - ant_image,
                           base,
                           base + ant_query(context, ANT_TLS_BYTES),
                           CORE_SIGNATURE};
}

static void exercise(unsigned tile) {
  size_t bytes = ant_image_end - ant_image;
  for (unsigned context = 0; context < 2; ++context) {
    if (bytes > ant_query(context, ANT_CODE_BYTES) ||
        ant_query(context, ANT_SIGNATURE) != CORE_SIGNATURE)
      __builtin_trap();
    ant_write(context, ANT_CODE, 0, ant_image, bytes);
  }
  for (unsigned i = 0; i < ROWS * ROW_WORDS; ++i) {
    input[tile][i] = ((uint64_t)(tile + 1) << 56) | (0x100 + i);
    output[tile][i] = 0;
  }
  uint64_t zero[2] = {0};
  ant_write(0, ANT_TSS, 0, zero, sizeof(zero));
  struct ant_mesh_move_args args = {(uintptr_t)input[tile],
                                    (uintptr_t)output[tile],
                                    ant_query(0, ANT_TSS_BASE), MESH_FILL};
  ant_acquire();
  run(0, tile * 3 + 1, &args);
  args.role = MESH_MOVE;
  struct ant_task move = prepare(0, tile * 3 + 2, &args);
  args.role = MESH_CONSUME;
  struct ant_task consume = prepare(1, tile * 3 + 3, &args);
  ant_start(1, &consume);
  ant_start(0, &move);
  int cancelled;
  if (ant_wait(0, move.id, &cancelled) || cancelled ||
      ant_wait(1, consume.id, &cancelled) || cancelled)
    __builtin_trap();
  ant_release();
  for (unsigned i = 0; i < ROWS * ROW_WORDS; ++i)
    if (output[tile][i] != input[tile][i])
      __builtin_trap();
  __atomic_store_n(&done[tile], 1, __ATOMIC_RELEASE);
}

int main(void) {
  uintptr_t hart;
  __asm__ volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart >= BB_MAIN_CORES)
    exercise(hart - BB_MAIN_CORES);
  else
    for (unsigned tile = 0; tile < BB_COMPUTE_TILES; ++tile)
      while (!__atomic_load_n(&done[tile], __ATOMIC_ACQUIRE)) {
      }
  return 0;
}
