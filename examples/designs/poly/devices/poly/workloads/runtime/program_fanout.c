#include <device.h>
#include <stdint.h>

enum { BYTES = 65553 };
static unsigned char source[BYTES];
static unsigned char slow[BYTES + 32] = {[0 ... BYTES + 31] = 0xa5};
static unsigned char fast[BYTES + 32] = {[0 ... BYTES + 31] = 0xa5};
// Hart 0 broadcasts to harts 1 and 2. Hart 1 cannot drain its link until
// hart 2 has received the complete value and finished its job.
static const uint64_t table[] = {3, 2, 3, 0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 1,
                                 0, 0, 1, 2, 1, 0, 1, 3, 0, 0, 0, 0, 3, 1,
                                 0, 0, 2, 0, 1, 3, 4, 1, 0, 0, 5, 1, 0, 0,
                                 0, 1, 0, 0, 2, 4, 2, 0, 1, 0, 1, 1};
static uint64_t words[8], state[3 * PROGRAM_STATE_WORDS];
static int send(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int receive_fast(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < BYTES; ++i)
    if (fast[16 + i] != (unsigned char)((i * 73 + (i >> 8)) ^ 0x35))
      return 1;
  for (unsigned i = 0; i < 16; ++i)
    if (fast[i] != 0xa5 || fast[16 + BYTES + i] != 0xa5)
      return 2;
  return 0;
}
static int receive_slow(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < BYTES; ++i)
    if (slow[16 + i] != (unsigned char)((i * 73 + (i >> 8)) ^ 0x35))
      return 1;
  for (unsigned i = 0; i < 16; ++i)
    if (slow[i] != 0xa5 || slow[16 + BYTES + i] != 0xa5)
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
  words[1] = (uintptr_t)(slow + 16);
  words[5] = (uintptr_t)(fast + 16);
  words[2] = words[6] = BYTES;
  words[3] = words[7] = 0;
  program_dispatch(0, send);
  program_dispatch(1, receive_slow);
  program_dispatch(2, receive_fast);
  program_finish();
  return 0;
}
int main(void) { return program_run(program, 0); }
