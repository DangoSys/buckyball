#include <device.h>
#include <stdint.h>

enum { BYTES = 65553 };
static unsigned char first[BYTES] = {[0 ... BYTES - 1] = 0x35};
static unsigned char second[BYTES] = {[0 ... BYTES - 1] = 0xa9};
static unsigned char early[BYTES] = {[0 ... BYTES - 1] = 0xa5};
static unsigned char late[BYTES] = {[0 ... BYTES - 1] = 0xa5};
// The two_sends compiler regression: two sends on hart 0, one reader on
// hart 1, which waits for the first send and drains both in FIFO order.
static const uint64_t joined_table[] = {
    4, 2, 2, 0, 3, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 2,  0,
    0, 0, 1, 1, 0, 1, 1, 0, 7, 0, 4, 2, 0, 2, 0, 2, 0, 2, 1, 1, 11, 1,
    4, 3, 0, 3, 0, 3, 2, 5, 0, 0, 1, 3, 0, 1, 7, 3, 0, 1, 0, 1};
static uint64_t joined_words[14], joined_state[4 * PROGRAM_STATE_WORDS];
static int send(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int receive(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return early[0] != 0x35 || early[BYTES / 2] != 0x35 ||
         early[BYTES - 1] != 0x35 || late[0] != 0xa9 ||
         late[BYTES / 2] != 0xa9 || late[BYTES - 1] != 0xa9;
}
static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  program_start(joined_table, joined_words, joined_state);
  joined_words[3] = (uintptr_t)first;
  joined_words[4] = (uintptr_t)early;
  joined_words[5] = BYTES;
  joined_words[7] = (uintptr_t)second;
  joined_words[8] = (uintptr_t)late;
  joined_words[9] = BYTES;
  joined_words[6] = 0;
  joined_words[10] = 1;
  program_dispatch(0, send);
  program_dispatch(1, send);
  program_dispatch(2, send);
  program_dispatch(3, receive);
  program_finish();
  return 0;
}
int main(void) { return program_run(program, 0); }
