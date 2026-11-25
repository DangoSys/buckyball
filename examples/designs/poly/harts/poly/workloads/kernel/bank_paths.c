#include <stdint.h>
#include <thread.h>

enum { WORDS = THREAD_MEM_BANK_ENTRIES / 4, CONTEXTS = BANK_PATH_THREADS };
#define FIRST UINT64_C(0x0706050403020100)
#define GUARD UINT64_C(0xa5a5a5a5a5a5a5a5)
static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "bank_paths.bin"
};
static uint64_t input[CONTEXTS][WORDS]
    __attribute__((aligned(4096))) = {{[0 ... WORDS - 1] = FIRST},
                                      {[0 ... WORDS - 1] = FIRST + 1},
                                      {[0 ... WORDS - 1] = FIRST + 2},
                                      {[0 ... WORDS - 1] = FIRST + 3},
                                      {[0 ... WORDS - 1] = FIRST + 4}};
static uint64_t output[CONTEXTS][WORDS + 16] __attribute__((aligned(4096))) = {
    [0 ... CONTEXTS - 1] = {[0 ... WORDS + 15] = GUARD}};

int main(void) {
  uint64_t shared = thread_query(0, THREAD_FIELD_TSS_BASE), zero = 0;
  struct thread_task tasks[CONTEXTS];
  thread_acquire();
  thread_tss_write(64, image, sizeof(image));
  for (unsigned i = 0; i < CONTEXTS; ++i) {
    uint32_t thread = i;
    input[i][0] = (FIRST + i) ^ (UINT64_C(1) << 63);
    input[i][WORDS - 1] = (FIRST + i) ^ (UINT64_C(1) << 62);
    *(volatile uint64_t *)&output[i][8] = GUARD;
    *(volatile uint64_t *)&output[i][WORDS + 7] = GUARD;
    uint64_t base = thread_query(thread, THREAD_FIELD_TLS_BASE);
    struct {
      uint64_t shared, publish, input, output;
    } args = {shared, i + 1 == CONTEXTS, (uintptr_t)input[i],
              (uintptr_t)(output[i] + 8)};
    thread_load(thread, THREAD_CODE, 0, 64, (sizeof(image) + 7) & ~(size_t)7);
    thread_tss_write(0, &args, sizeof(args));
    thread_load(thread, THREAD_TLS, 0, 0, sizeof(args));
    tasks[i] = (struct thread_task){
        .id = i + 1,
        .entry = 0,
        .code_end = sizeof(image),
        .argument = base,
        .stack = base + thread_query(thread, THREAD_FIELD_TLS_BYTES),
        .signature = thread_query(thread, THREAD_FIELD_SIGNATURE)};
  }
  thread_tss_write(0, &zero, sizeof(zero));
  for (unsigned i = 0; i < CONTEXTS; ++i)
    thread_start(i, &tasks[i]);
  for (unsigned i = 0; i < CONTEXTS; ++i) {
    int cancelled;
    if (thread_wait(i, tasks[i].id, &cancelled) || cancelled)
      return 1;
  }
  thread_release();
  // The final bank read is checked against BEMU; keep CPU checks to bounds.
  for (unsigned i = 0; i < CONTEXTS; ++i) {
    uint64_t expected = FIRST + i;
    if (output[i][8] != (expected ^ (UINT64_C(1) << 63)) ||
        output[i][WORDS + 7] != (expected ^ (UINT64_C(1) << 62)))
      return 2;
    for (unsigned j = 0; j < 8; ++j)
      if (output[i][j] != GUARD || output[i][WORDS + 8 + j] != GUARD)
        return 3;
  }
  return 0;
}
