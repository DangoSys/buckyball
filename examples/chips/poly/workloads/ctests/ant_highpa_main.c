#include "ant_highpa.h"
#include <ant.h>
#include <params.h>
#include <topology.h>
extern const unsigned char ant_image[], ant_image_end[];
static unsigned done[BB_COMPUTE_TILES];
static void exercise(unsigned tile) {
  const unsigned context = BB_ANT_CONTEXT;
  uintptr_t high = 0x180200000ULL + tile * 0x20000ULL;
  volatile uint64_t *input = (void *)high, *output = (void *)(high + 0x1000);
  volatile uint64_t *low = (void *)(high & 0xffffffffULL);
  volatile uint64_t *sentinel = (void *)((high + 0x1000) & 0xffffffffULL);
  const uint64_t marker = 0xdeadface12345678ULL;
  uint64_t first = 0x1234567800000000ULL | ((uint64_t)tile << 16);
  uint64_t second = 0xfedcba9800000000ULL | ((uint64_t)tile << 16);
  for (unsigned i = 0; i < 4; ++i) {
    low[i] = second + i;
    sentinel[i] = marker;
    input[i] = first + i;
    output[i] = 0;
  }
  for (unsigned i = 0; i < 4; ++i)
    if (input[i] != first + i || low[i] != second + i || sentinel[i] != marker)
      __builtin_trap();
  size_t bytes = ant_image_end - ant_image;
  if (bytes > ant_query(context, ANT_CODE_BYTES) ||
      ant_query(context, ANT_SIGNATURE) != CORE_SIGNATURE)
    __builtin_trap();
  ant_write(context, ANT_CODE, 0, ant_image, bytes);
  uint64_t base = ant_query(context, ANT_TLS_BASE);
  for (unsigned phase = 0; phase < 2; ++phase) {
    struct ant_highpa_args args = {phase ? (uintptr_t)low : high,
                                   high + 0x1000};
    ant_write(context, ANT_TLS, 0, &args, sizeof(args));
    struct ant_task task = {tile * 2 + phase + 1,
                            0,
                            bytes,
                            base,
                            base + ant_query(context, ANT_TLS_BYTES),
                            CORE_SIGNATURE};
    ant_acquire();
    ant_start(context, &task);
    int cancelled;
    if (ant_wait(context, task.id, &cancelled) || cancelled)
      __builtin_trap();
    ant_release();
    for (unsigned i = 0; i < 4; ++i)
      if (output[i] != (phase ? second : first) + i || sentinel[i] != marker ||
          input[i] != first + i || low[i] != second + i)
        __builtin_trap();
  }
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
