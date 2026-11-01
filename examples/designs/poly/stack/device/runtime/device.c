/* The poly device runtime: values between cores over the core links.
 *
 * Every core has a communication storage of its own; its CPU writes and
 * reads it a 64-bit word at a time, and moves bytes from it into the storage
 * of a core it has a route to (core 0 and every poly core, both ways).
 * A value crosses a link in chunks: a push job on the sending core writes a
 * chunk from its buffer into its storage and moves it into the storage of
 * the receiving core, a pull job on the receiving core reads it into the
 * receiving buffer.
 *
 * The storage of a core is halved: the low half stages what the core sends,
 * the high half has one slot per sending core. The chunks of one link go
 * through its slot one after the other: a push waits for the pull of the
 * chunk before it on the same link. The jobs of a core run one at a time, so
 * the staging half is never shared. */
#include "device.h"

#include <string.h>

#define X(hart) +1
enum { CORE_COUNT = 0 DEVICE_CORES };
#undef X

#define LINK(op, field, value)                                                 \
  ({                                                                           \
    uint64_t result;                                                           \
    asm volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2"                        \
                 : "=r"(result)                                                \
                 : "r"((uint64_t)(field)), "r"((uint64_t)(value))              \
                 : "memory");                                                  \
    result;                                                                    \
  })
#define LINK_CAPACITY 0

/* The job of the last pull of every link and its run: a pull of an earlier
 * run is done before this run starts. */
static uint64_t pulled[CORE_COUNT][CORE_COUNT];
static uint64_t pulled_run[CORE_COUNT][CORE_COUNT];

static int push(void *core, void *argument) {
  (void)core;
  const uint64_t *words = argument;
  const uint8_t *source = (const uint8_t *)(uintptr_t)words[0];
  const uint64_t bytes = words[1];
  for (uint64_t offset = 0; offset < bytes; offset += 8) {
    uint64_t word = 0;
    memcpy(&word, source + offset, bytes - offset < 8 ? bytes - offset : 8);
    if (LINK(17, offset, word))
      __builtin_trap();
  }
  LINK(13, 0, 0);
  LINK(13, 1, words[2]);
  LINK(13, 2, words[3]);
  LINK(13, 3, (bytes + 7) & ~(uint64_t)7);
  if (LINK(14, 0, 0))
    __builtin_trap();
  return 0;
}

static int pull(void *core, void *argument) {
  (void)core;
  const uint64_t *words = argument;
  uint8_t *target = (uint8_t *)(uintptr_t)words[0];
  const uint64_t bytes = words[1];
  for (uint64_t offset = 0; offset < bytes; offset += 8) {
    const uint64_t word = LINK(16, words[2] + offset, 0);
    memcpy(target + offset, &word, bytes - offset < 8 ? bytes - offset : 8);
  }
  return 0;
}

uint64_t device_transfer(uint64_t from, uint64_t to, uint64_t target,
                         uint64_t source, uint64_t bytes, uint64_t after) {
  // The slot of a sending core: the high half of the storage in a slot per
  // core, 16-byte aligned. Every core's storage has the capacity of the
  // program's.
  const uint64_t half = LINK(15, LINK_CAPACITY, 0) / 2;
  const uint64_t slot = half / CORE_COUNT & ~(uint64_t)15;
  const uint64_t base = half + from * slot;
  uint64_t job = after;
  for (uint64_t offset = 0; offset < bytes; offset += slot) {
    const uint64_t chunk = bytes - offset < slot ? bytes - offset : slot;
    program_job_begin(from, push);
    program_job_argument(source + offset);
    program_job_argument(chunk);
    program_job_argument(to);
    program_job_argument(base);
    program_job_after(after);
    if (pulled_run[from][to] == program_run_index())
      program_job_after(pulled[from][to]);
    const uint64_t pushed = program_job_submit();
    program_job_begin(to, pull);
    program_job_argument(target + offset);
    program_job_argument(chunk);
    program_job_argument(base);
    program_job_after(pushed);
    job = program_job_submit();
    pulled[from][to] = job;
    pulled_run[from][to] = program_run_index();
  }
  return job;
}
