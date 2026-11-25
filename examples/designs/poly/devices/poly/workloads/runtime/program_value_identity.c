#include <device.h>
#include <stdint.h>

enum { BYTES = 65553 };
static unsigned char source[BYTES];
static unsigned char first[BYTES + 32] = {[0 ... BYTES + 31] = 0xa5};
static unsigned char second[BYTES + 32] = {[0 ... BYTES + 31] = 0xa5};
// Two logical values alias the same DDR bytes. Their frames still reach the
// receiving hart in order, rather than interleaving their chunks.
static const uint64_t table[] = {2, 2, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0,
                                 0, 1, 2, 1, 0, 1, 2, 0, 0, 0, 0, 3, 2, 0,
                                 0, 0, 1, 0, 0, 1, 4, 1, 0, 1, 0, 1};
static uint64_t words[8], state[2 * PROGRAM_STATE_WORDS];
static int send(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int receive(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < BYTES; ++i) {
    const unsigned char expected = (unsigned char)((i * 73 + (i >> 8)) ^ 0x35);
    if (first[16 + i] != expected || second[16 + i] != expected)
      return 1;
  }
  for (unsigned i = 0; i < 16; ++i)
    if (first[i] != 0xa5 || second[i] != 0xa5 ||
        first[16 + BYTES + i] != 0xa5 || second[16 + BYTES + i] != 0xa5)
      return 2;
  return 0;
}
static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < BYTES; ++i)
    source[i] = (unsigned char)((i * 73 + (i >> 8)) ^ 0x35);
  program_start(table, words, state);
  words[0] = words[4] = (uintptr_t)source;
  words[1] = (uintptr_t)(first + 16);
  words[5] = (uintptr_t)(second + 16);
  words[2] = words[6] = BYTES;
  words[3] = 0;
  words[7] = 1;
  program_dispatch(0, send);
  program_dispatch(1, receive);
  program_finish();
  return 0;
}
int main(void) { return program_run(program, 0); }
