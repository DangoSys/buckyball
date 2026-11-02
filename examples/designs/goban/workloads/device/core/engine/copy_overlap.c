#include "copy.h"
#include <engine.h>
#include <stdio.h>

enum { ROWS = 512, BYTES = ROWS * (THREAD_MEM_BANK_WIDTH / 8) };
_Static_assert(THREAD_MEM_BANK_ENTRIES >= ROWS,
               "Copy overlap requires 512 bank rows");
static const uint64_t input[4][BYTES / 8 + 4] __attribute__((aligned(4096))) = {
    {[0 ... BYTES / 8 + 3] = UINT64_C(0x0123456789abcdef)},
    {[0 ... BYTES / 8 + 3] = UINT64_C(0x23456789abcdef01)},
    {[0 ... BYTES / 8 + 3] = UINT64_C(0x456789abcdef0123)},
    {[0 ... BYTES / 8 + 3] = UINT64_C(0x6789abcdef012345)}};
static volatile uint64_t output[4][BYTES / 8 + 4] __attribute__((
    aligned(4096))) = {{[0 ... BYTES / 8 + 3] = UINT64_C(0xa5a5a5a5a5a5a5a5)},
                       {[0 ... BYTES / 8 + 3] = UINT64_C(0xa5a5a5a5a5a5a5a5)},
                       {[0 ... BYTES / 8 + 3] = UINT64_C(0xa5a5a5a5a5a5a5a5)},
                       {[0 ... BYTES / 8 + 3] = UINT64_C(0xa5a5a5a5a5a5a5a5)}};

static uint64_t cycle(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value)::"memory");
  return value;
}

int main(void) {
  struct engine engine;
  struct engine_slot slots[4];
  struct copy_task args[4];
  struct engine_job jobs[4];
  struct engine_result results[4];
  engine_init(&engine, slots, 4);
  const unsigned sources[] = {0, 4095, 4081};
  const unsigned destinations[] = {0, 4081, 4095};
  for (unsigned test = 0; test < 3; ++test) {
    unsigned source = sources[test];
    unsigned destination = destinations[test];
    unsigned rows = test == 0 ? ROWS : 8;
    unsigned bytes = rows * (THREAD_MEM_BANK_WIDTH / 8);
    unsigned first = destination == 0 ? 0 : destination / 8 - 1;
    unsigned end = (destination + bytes + 7) / 8 + 1;
    for (unsigned thread = 0; thread < 4; ++thread) {
      if (test != 0)
        for (unsigned word = first; word < end; ++word)
          output[thread][word] = UINT64_C(0xa5a5a5a5a5a5a5a5);
      args[thread] =
          (struct copy_task){(uintptr_t)input[thread] + source,
                             (uintptr_t)output[thread] + destination, rows};
      jobs[thread] = (struct engine_job){
          .thread = thread,
          .task = {.id = thread + 1, .signature = slots[thread].signature},
          .opaque = &args[thread]};
    }
    uint64_t start = cycle();
    if (engine_run(&engine, jobs, 4, 0, 0, results))
      return 1;
    uint64_t elapsed = cycle() - start;
    printf("Core copy overlap (%u -> %u): %u cycles\n", source, destination,
           (unsigned)elapsed);
    for (unsigned thread = 0; thread < 4; ++thread) {
      if (results[thread].state != ENGINE_SUCCEEDED || results[thread].value)
        return 2;
      for (unsigned word = first; word < end; ++word) {
        uint64_t expected = UINT64_C(0xa5a5a5a5a5a5a5a5);
        if (test == 0 && word < BYTES / 8) {
          expected = input[thread][word];
        } else if (test != 0 && word * 8 < destination + bytes &&
                   word * 8 + 8 > destination) {
          for (unsigned lane = 0; lane < 8; ++lane) {
            unsigned byte = word * 8 + lane;
            if (byte >= destination && byte < destination + bytes) {
              uint64_t value =
                  ((const uint8_t *)input[thread])[source + byte - destination];
              expected = (expected & ~(UINT64_C(0xff) << (lane * 8))) |
                         (value << (lane * 8));
            }
          }
        }
        if (output[thread][word] != expected)
          return 3;
      }
    }
  }
  return 0;
}
