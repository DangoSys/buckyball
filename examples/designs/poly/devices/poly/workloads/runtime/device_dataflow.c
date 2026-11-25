#include "copy.h"
#include <device.h>
#include <kernel.h>
#include <stdint.h>

enum { BYTES = 16400 };
static const unsigned char source[BYTES + 8] = {
    [0 ... BYTES + 7] = 0x3c, [3] = 0x11, [4099] = 0x22, [16387] = 0x55};
static unsigned char target[BYTES + 16] = {[0 ... BYTES + 15] = 0xa5};
static unsigned char output[BYTES]
    __attribute__((aligned(64), section(".noinit")));
static const uint64_t table[] = {2, 1, 2, 0, 1, 0, 0, 0, 2, 0, 0, 0,
                                 1, 0, 0, 1, 1, 1, 0, 1, 2, 0, 0, 0,
                                 0, 2, 1, 0, 0, 0, 1, 0, 1, 0, 0};
static uint64_t words[4], state[2 * PROGRAM_STATE_WORDS];
static int job(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  return 0;
}
static int consume(void *hart, void *argument) {
  (void)argument;
  struct kernel *owner = hart;
  struct copy_task args[2] = {
      {(uintptr_t)(target + 5), (uintptr_t)output, BYTES / 16},
      {(uintptr_t)output, (uintptr_t)output, BYTES / 16}};
  struct kernel_job tasks[2] = {0};
  struct kernel_result results[2];
  const size_t dependency = 0;
  for (unsigned i = 0; i < 2; ++i) {
    tasks[i].task.id = i + 1;
    tasks[i].task.signature = owner->slots[0].signature;
    tasks[i].opaque = &args[i];
  }
  tasks[1].dependencies = &dependency;
  tasks[1].dependency_count = 1;
  // Readback exposes every output byte to Bank DiffTest, including dirty L2
  // lines.
  if (kernel_run(owner, tasks, 2, 0, 0, results))
    return 1;
  if (target[5] != 0x11 || target[4101] != 0x22 || target[16389] != 0x55)
    return 4;
  for (unsigned i = 0; i < 5; ++i)
    if (target[i] != 0xa5)
      return 2;
  for (unsigned i = BYTES + 5; i < BYTES + 16; ++i)
    if (target[i] != 0xa5)
      return 3;
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
  program_dispatch(0, job);
  program_dispatch(1, consume);
  program_finish();
  return 0;
}
int main(void) { return program_run(program, 0); }
