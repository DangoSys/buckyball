#include <device.h>
#include <stdint.h>
#include <stdio.h>

enum { BYTES = 196625 };
static unsigned char source[BYTES], other[BYTES];
static unsigned char joined[2][BYTES + 32], broadcast[BYTES + 32];
// Hart 2 has two senders, hart 3 only one: their receive rings differ.
// Every value exceeds both ring capacities and must stream under backpressure.
static uint64_t table[] = {
    4, 3, 4, 0, 1, 2, 3, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 2, 2, 1, 0, 0, 4, 0, 0,
    4, 1, 0, 0, 5, 1, 2, 0, 2, 4, 0, 0, 0, 0, 6, 2, 0, 0, 3, 0, 1, 4, 0, 0, 0,
    0, 8, 1, 0, 0, 0, 2, 0, 0, 3, 4, 1, 2, 8, 2, 3, 0, 1, 2, 2, 0, 2, 1};
static uint64_t words[12], state[4 * PROGRAM_STATE_WORDS];
static unsigned run;
static int send(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int verify(void) {
  for (unsigned i = 0; i < BYTES; ++i) {
    const unsigned char expected = (unsigned char)(i * 73 + run);
    if (joined[0][i + 16] != expected ||
        joined[1][i + 16] != (unsigned char)(expected ^ 0xa9))
      return 1;
    if (broadcast[i + 16] != expected)
      return 2;
  }
  for (unsigned i = 0; i < 16; ++i)
    if (joined[0][i] || joined[0][BYTES + 16 + i] || joined[1][i] ||
        joined[1][BYTES + 16 + i] || broadcast[i] || broadcast[BYTES + 16 + i])
      return 3;
  return 0;
}
static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  ++run;
  if (run == 2) {
    // Change the incoming topology, reusing storage with live old headers.
    table[5] = 3;
    table[6] = 2;
    uint64_t *entries = table + 7;
    entries[2 * PROGRAM_ENTRY_WORDS] = 3;
    entries[3 * PROGRAM_ENTRY_WORDS] = 2;
    struct program_transfer *moved =
        (struct program_transfer *)(entries + 4 * PROGRAM_ENTRY_WORDS);
    moved[0].to = moved[2].to = 3;
    moved[1].to = 2;
  }
  for (unsigned i = 0; i < BYTES; ++i) {
    source[i] = (unsigned char)(i * 73 + run);
    other[i] = source[i] ^ 0xa9;
  }
  program_start(table, words, state);
  words[0] = words[4] = (uintptr_t)source;
  words[8] = (uintptr_t)other;
  words[1] = (uintptr_t)(joined[0] + 16);
  words[5] = (uintptr_t)(broadcast + 16);
  words[9] = (uintptr_t)(joined[1] + 16);
  words[2] = words[6] = words[10] = BYTES;
  words[3] = words[7] = 0;
  words[11] = 1;
  program_dispatch(0, send);
  program_dispatch(1, send);
  program_dispatch(2, send);
  program_dispatch(3, send);
  program_finish();
  return 0;
}
int main(void) {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  for (unsigned i = 0; i < 2; ++i) {
    if (program_run(program, 0) || (hart == DEVICE_CONTROL_HART && verify()))
      return 1;
  }
  if (hart == DEVICE_CONTROL_HART)
    printf("Device topology cycles: %lu\n", (unsigned long)program_latency());
  return 0;
}
