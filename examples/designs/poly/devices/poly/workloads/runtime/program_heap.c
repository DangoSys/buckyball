#include <device.h>
#include <stdint.h>
#include <stdlib.h>

#define X(hart) +1
enum { HARTS = 0 DEVICE_HARTS };
#undef X
static uint64_t table[3 + HARTS + HARTS * PROGRAM_ENTRY_WORDS];
static uint64_t words[HARTS], state[HARTS * PROGRAM_STATE_WORDS];
extern void __malloc_lock(void *);
extern void __malloc_unlock(void *);

static int allocate(void *hart, void *argument) {
  (void)hart;
  const uint64_t rank = *(uint64_t *)argument;
  for (unsigned round = 0; round < 4; ++round) {
    const size_t count = 8 + rank + round;
    __malloc_lock(0);
    uint64_t *first = malloc(count * sizeof(uint64_t));
    uint64_t *second = malloc(count * sizeof(uint64_t));
    for (unsigned wait = 0; wait < 16; ++wait)
      asm volatile("nop");
    __malloc_unlock(0);
    for (size_t i = 0; i < count; ++i) {
      first[i] = (rank << 32) + round * 256 + i;
      second[i] = ~first[i];
    }
    for (size_t i = 0; i < count; ++i)
      if (first[i] != (rank << 32) + round * 256 + i || second[i] != ~first[i])
        return 1;
    free(second);
    free(first);
  }
  return 0;
}

static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  table[0] = table[2] = HARTS;
  for (unsigned h = 0; h < HARTS; ++h) {
    table[3 + h] = h;
    uint64_t *entry = table + 3 + HARTS + h * PROGRAM_ENTRY_WORDS;
    entry[0] = entry[1] = words[h] = h;
    entry[3] = HARTS;
  }
  program_start(table, words, state);
  for (unsigned h = 0; h < HARTS; ++h)
    program_dispatch(h, allocate);
  program_finish();
  return 0;
}

int main(void) {
  void *initial = malloc(64);
  asm volatile("" ::"r"(initial) : "memory");
  free(initial);
  for (unsigned run = 0; run < 2; ++run)
    if (program_run(program, 0))
      return 1;
  return 0;
}
