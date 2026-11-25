#include <device.h>
#include <stdint.h>

static const uint64_t table[] = {1, 0, 1, 0, 0, 0, 0, 1,
                                 0, 0, 0, 0, 0, 0, 0, 0};
static uint64_t words[1], state[PROGRAM_STATE_WORDS], measured;
static volatile uint64_t value;
static int compute(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  for (unsigned i = 0; i < 512; ++i)
    value += i;
  return 0;
}
static int program(void *hart, void *argument) {
  (void)hart;
  (void)argument;
  uint64_t first, last;
  asm volatile("rdcycle %0" : "=r"(first) : : "memory");
  program_start(table, words, state);
  program_dispatch(0, compute);
  program_finish();
  asm volatile("rdcycle %0" : "=r"(last) : : "memory");
  measured = last - first;
  return 0;
}
int main(void) {
  int status = program_run(program, 0);
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (status || hart != DEVICE_CONTROL_HART)
    return status;
  // A single CPU job cannot spend more modeled time than the whole call.
  return program_latency() > measured || program_busy(hart) > measured;
}
