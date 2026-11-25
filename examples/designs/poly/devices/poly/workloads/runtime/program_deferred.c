#include <device.h>
#include <stdint.h>

enum { BYTES = 65553 };
static unsigned char first[BYTES] = {[0 ... BYTES - 1] = 0x35};
static unsigned char second[BYTES] = {[0 ... BYTES - 1] = 0xa9};
static unsigned char early[BYTES] = {[0 ... BYTES - 1] = 0xa5};
static unsigned char late[BYTES] = {[0 ... BYTES - 1] = 0xa5};
// Produce A, receive A, send B later on the producer hart, then receive B.
// Each value exceeds the link's ring capacity.
static const uint64_t table[] = {
    4, 2, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 3, 0,
    0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 4, 0, 0, 3, 1, 0, 0, 4, 1, 1, 0, 1,
    4, 0, 0, 0, 0, 5, 1, 0, 0, 0, 1, 0, 0, 1, 4, 1, 0, 0, 3, 1, 1};
static uint64_t words[8], state[4 * PROGRAM_STATE_WORDS];
static int produce(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  first[0] = 0x12;
  first[BYTES - 1] = 0x34;
  second[0] = 0x56;
  second[BYTES - 1] = 0x78;
  return 0;
}
static int receive_first(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return early[0] != 0x12 || early[BYTES / 2] != 0x35 ||
         early[BYTES - 1] != 0x34;
}
static int send_later(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int receive_second(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return late[0] != 0x56 || late[BYTES / 2] != 0xa9 || late[BYTES - 1] != 0x78;
}
static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  program_start(table, words, state);
  words[0] = (uintptr_t)first;
  words[1] = (uintptr_t)early;
  words[2] = BYTES;
  words[4] = (uintptr_t)second;
  words[5] = (uintptr_t)late;
  words[6] = BYTES;
  words[3] = 0;
  words[7] = 1;
  program_dispatch(0, produce);
  program_dispatch(1, receive_first);
  program_dispatch(2, send_later);
  program_dispatch(3, receive_second);
  program_finish();
  return 0;
}
int main(void) { return program_run(program, 0); }
