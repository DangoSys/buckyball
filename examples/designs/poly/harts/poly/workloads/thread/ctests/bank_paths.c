#include <bbhw/isa/isa.h>
#include <stdint.h>

struct bank_paths_args {
  uint64_t shared, publish, input, output;
};
uint64_t entry(const struct bank_paths_args *args) {
  volatile uint64_t *ready = (volatile uint64_t *)args->shared;
  if (args->publish)
    *ready = 1;
  else
    while (!*ready) {
    }
  __asm__ volatile("fence rw, rw" ::: "memory");
  bb_mem_alloc(1, 1, 1);
  bb_mvin(args->input, 1, THREAD_MEM_BANK_ENTRIES / 8, 1);
  bb_mvout(args->output, 1, THREAD_MEM_BANK_ENTRIES / 8, 1);
  __asm__ volatile("fence" ::: "memory");
  // Read the output through coherence; BankDiff checks every returned row.
  bb_mvin(args->output, 1, THREAD_MEM_BANK_ENTRIES / 8, 1);
  bb_mem_release(1);
  __asm__ volatile("fence" ::: "memory");
  return 0;
}
