#include <device.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { BYTES = 65553 };
static unsigned char *source, *target;
static unsigned run;

static int produce(void *core, void *argument) {
  (void)core;
  (void)argument;
  for (unsigned i = 0; i < BYTES; ++i)
    source[i + 3] = (unsigned char)(i * 17 + run);
  return 0;
}

static int consume(void *core, void *argument) {
  (void)core;
  (void)argument;
  for (unsigned i = 0; i < BYTES + 16; ++i) {
    const unsigned char expected =
        i >= 5 && i < BYTES + 5 ? (unsigned char)((i - 5) * 17 + run) : 0xa5;
    if (target[i] != expected)
      return 1;
  }
  return 0;
}

static int submit(void *core, void *argument) {
  (void)core;
  (void)argument;
  program_job_begin(0, produce);
  const uint64_t produced = program_job_submit();
  const uint64_t copied = device_transfer(
      0, 1, (uintptr_t)(target + 5), (uintptr_t)(source + 3), BYTES, produced);
  program_job_begin(1, consume);
  program_job_after(copied);
  program_job_submit();
  program_cleanup_begin(2);
  program_job_argument((uintptr_t)source);
  program_job_argument((uintptr_t)target);
  program_job_submit();
  return 0;
}

int main(void) {
  for (run = 1; run <= 2; ++run) {
    source = malloc(BYTES + 8);
    target = malloc(BYTES + 16);
    memset(target, 0xa5, BYTES + 16);
    if (program_run(submit, 0))
      return 1;
  }
  return 0;
}
