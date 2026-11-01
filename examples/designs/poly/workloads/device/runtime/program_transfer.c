#include <device.h>
#include <stdint.h>
#include <string.h>

enum { BYTES = 65553 };
static unsigned char source[BYTES + 8], target[BYTES + 16];
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
  for (unsigned i = 0; i < sizeof(target); ++i) {
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
  return 0;
}

int main(void) {
  for (run = 1; run <= 2; ++run) {
    memset(target, 0xa5, sizeof(target));
    if (program_run(submit, 0))
      return 1;
  }
  return 0;
}
