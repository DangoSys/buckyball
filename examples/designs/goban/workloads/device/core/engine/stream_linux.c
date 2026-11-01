#include <bb_isa.h>
#include <engine.h>
#include <linux.h>
#include <stream.h>

enum { rows = 8, words = rows * THREAD_MEM_BANK_WIDTH / 64 };
static uint64_t input[words] __attribute__((aligned(64)));
static uint64_t intermediate[2][words] __attribute__((aligned(64)));
static uint64_t output[words] __attribute__((aligned(64)));

// Record the same bank instructions emitted by the compiler's lower-acc.
static void copy(void *argument) {
  const uintptr_t *addresses = argument;
  stream_emit(BB_MSET_FUNC7, BB_BANK0(1),
              FIELD(1, 0, 4) | FIELD(1, 5, 9) | FIELD(1, 10, 10));
  stream_emit(BB_MVIN_FUNC7, BB_BANK2(1) | BB_ITER(rows),
              FIELD(addresses[0], 0, 38) | FIELD(1, 39, 57));
  stream_emit(BB_MVOUT_FUNC7, BB_BANK0(1) | BB_ITER(rows),
              FIELD(addresses[1], 0, 38) | FIELD(1, 39, 57));
  stream_emit(BB_MSET_FUNC7, BB_BANK0(1), 0);
}

static int compute(void *context, void *argument) {
  (void)argument;
  struct engine *engine = context;
  if (engine->thread_count < 2)
    return 1;
  for (unsigned iteration = 0; iteration < 3; ++iteration) {
    for (unsigned word = 0; word < words; ++word) {
      input[word] = ((uint64_t)(iteration + 1) << 32) | word;
      output[word] = 0;
      intermediate[0][word] = intermediate[1][word] = UINT64_MAX;
    }
    uintptr_t first[] = {(uintptr_t)input, (uintptr_t)intermediate[0]};
    uintptr_t second[] = {(uintptr_t)intermediate[1], (uintptr_t)output};
    stream_begin(2, 1, 1);
    stream_job(engine->slots[0].signature, 0, copy, first);
    stream_job(engine->slots[1].signature, 1, copy, second);
    stream_after(0);
    stream_transfer(0, 1, intermediate[0], intermediate[1], sizeof(input));
    stream_submit();
    stream_wait();
    for (unsigned word = 0; word < words; ++word) {
      if (output[word] != input[word])
        return 2;
      // The core's hidden storage carries the edge, leaving DDR untouched.
      if (intermediate[0][word] != UINT64_MAX ||
          intermediate[1][word] != UINT64_MAX)
        return 3;
    }
  }
  return 0;
}

int main(void) {
  // A second worker reloads and reuses the same hardware after the first exits.
  for (unsigned run = 0; run < 2; ++run) {
    int status = runtime_linux_job(0, DEVICE_CORES_0_SIGNATURE, compute, 0);
    if (status)
      return status;
  }
  return 0;
}
