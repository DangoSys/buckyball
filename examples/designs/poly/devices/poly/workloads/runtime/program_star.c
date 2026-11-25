#include <device.h>
#include <stdint.h>
#include <stdio.h>

#define X(hart) +1
enum {
  HARTS = 0 DEVICE_HARTS,
  PEERS = HARTS - 1,
  JOBS = HARTS + 1,
  BYTES = 196625,
  LISTS = 6 * PEERS
};
#undef X
static unsigned char source[BYTES], local[PEERS][BYTES + 32],
    result[PEERS][BYTES + 32];
static uint64_t
    table[3 + HARTS + JOBS * PROGRAM_ENTRY_WORDS + 6 * PEERS + LISTS];
static uint64_t words[8 * PEERS], state[JOBS * PROGRAM_STATE_WORDS];
static int job(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  table[0] = JOBS;
  table[1] = 2 * PEERS;
  table[2] = HARTS;
  uint64_t *entries = table + 3 + HARTS;
  struct program_transfer *moves =
      (struct program_transfer *)(entries + JOBS * PROGRAM_ENTRY_WORDS);
  uint64_t *lists = (uint64_t *)(moves + 2 * PEERS);
  for (unsigned h = 0; h < HARTS; ++h)
    table[3 + h] = h;
  entries[3] = HARTS;
  entries[6] = 0;
  entries[7] = entries[11] = PEERS;
  entries[10] = PEERS;
  uint64_t *last = entries + HARTS * PROGRAM_ENTRY_WORDS;
  last[2] = last[9] = PEERS;
  last[3] = JOBS;
  last[8] = 5 * PEERS;
  for (unsigned p = 0; p < PEERS; ++p) {
    const unsigned h = p + 1;
    uint64_t *entry = entries + h * PROGRAM_ENTRY_WORDS;
    entry[0] = h;
    entry[2] = entry[7] = entry[9] = entry[11] = 1;
    entry[3] = JOBS;
    entry[6] = 2 * PEERS + 3 * p;
    entry[8] = entry[6] + 1;
    entry[10] = entry[6] + 2;
    lists[p] = h;
    lists[PEERS + p] = p;
    lists[2 * PEERS + 3 * p] = HARTS;
    lists[2 * PEERS + 3 * p + 1] = p;
    lists[2 * PEERS + 3 * p + 2] = PEERS + p;
    lists[5 * PEERS + p] = PEERS + p;
    moves[p] = (struct program_transfer){0, h, 4 * p};
    moves[PEERS + p] = (struct program_transfer){h, 0, 4 * (PEERS + p)};
    words[4 * p] = (uintptr_t)source;
    words[4 * p + 1] = (uintptr_t)(local[p] + 16);
    words[4 * p + 2] = words[4 * (PEERS + p) + 2] = BYTES;
    words[4 * p + 3] = 0;
    words[4 * (PEERS + p)] = (uintptr_t)(local[p] + 16);
    words[4 * (PEERS + p) + 1] = (uintptr_t)(result[p] + 16);
    words[4 * (PEERS + p) + 3] = p + 1;
  }
  for (unsigned i = 0; i < BYTES; ++i)
    source[i] = (unsigned char)(i * 73 + 9);
  program_start(table, words, state);
  for (unsigned i = 0; i < JOBS; ++i)
    program_dispatch(i, job);
  program_finish();
  return 0;
}
int main(void) {
  if (program_run(program, 0))
    return 1;
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (hart != DEVICE_CONTROL_HART)
    return 0;
  for (unsigned p = 0; p < PEERS; ++p) {
    for (unsigned i = 0; i < BYTES; ++i)
      if (local[p][i + 16] != source[i] || result[p][i + 16] != source[i])
        return 2;
    for (unsigned i = 0; i < 16; ++i)
      if (local[p][i] || local[p][BYTES + 16 + i] || result[p][i] ||
          result[p][BYTES + 16 + i])
        return 3;
  }
  printf("Device star cycles: %lu\n", (unsigned long)program_latency());
  return 0;
}
