#include <device.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Hart 0 writes a value that hart 1 reads: an unaligned source and target,
// more chunks than a link has parts.
enum { BYTES = 65553 };
static unsigned char *source, *target;
static unsigned run;
// Two entries: 0 on hart 0 pushes transfer 0, 1 on hart 1 pulls it.
static const uint64_t table[] = {2, 1, 2, 0, 1, // counts, the first entries
                                 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 1, 1, // entry 0
                                 1, 0, 1, 2, 0, 0, 0, 0, 2, 1, 0, 0, // entry 1
                                 0, 1, 0,  // transfer 0
                                 1, 0, 0}; // lists
static uint64_t words[4], state[2 * PROGRAM_STATE_WORDS];

static int produce(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < BYTES; ++i)
    source[i + 3] = (unsigned char)(i * 17 + run);
  return 0;
}

static int consume(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < BYTES + 16; ++i) {
    const unsigned char expected =
        i >= 5 && i < BYTES + 5 ? (unsigned char)((i - 5) * 17 + run) : 0xa5;
    if (target[i] != expected)
      return 1;
  }
  return 0;
}

static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  program_start(table, words, state);
  words[0] = (uintptr_t)(source + 3);
  words[1] = (uintptr_t)(target + 5);
  words[2] = BYTES;
  words[3] = 0;
  program_dispatch(0, produce);
  program_dispatch(1, consume);
  program_finish();
  return 0;
}

int main(void) {
  for (run = 1; run <= 2; ++run) {
    source = malloc(BYTES + 8);
    target = malloc(BYTES + 16);
    memset(target, 0xa5, BYTES + 16);
    if (program_run(program, 0))
      return 1;
    free(source);
    free(target);
  }
  return 0;
}
