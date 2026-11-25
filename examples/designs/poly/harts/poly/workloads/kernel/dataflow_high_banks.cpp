#include "copy.h"
#include <array>
#include <kernel.h>

enum { HUGE = 1 << 20, WORDS = 4098, BYTES = WORDS * 8 };
alignas(16) static constexpr auto input = [] {
  std::array<uint64_t, WORDS> data{};
  for (unsigned i = 0; i < WORDS; ++i)
    data[i] = UINT64_C(0x1234567800000000) + i * 17;
  return data;
}();
alignas(16) static const uint64_t prefix[] = {0x1122334455667788,
                                              0x8877665544332211};
alignas(16) static uint64_t reserved[HUGE / 8]
    __attribute__((section(".noinit")));
alignas(16) static uint64_t temporary[WORDS]
    __attribute__((section(".noinit")));
alignas(16) static uint64_t output[WORDS] __attribute__((section(".noinit")));
alignas(16) static uint64_t small[2] __attribute__((section(".noinit")));
static unsigned done;
static kernel_job jobs[5];
static kernel_transfer edges[2];
static const copy_task arguments[] = {
    {(uintptr_t)prefix, (uintptr_t)reserved, 1},
    {(uintptr_t)input.data(), (uintptr_t)temporary, BYTES / 16},
    {(uintptr_t)temporary, (uintptr_t)output, BYTES / 16},
    {(uintptr_t)output, (uintptr_t)output, BYTES / 16},
    {(uintptr_t)reserved, (uintptr_t)small, 1}};
__attribute__((section(".gvn.bank_output"),
               used)) static const uint64_t check[] = {
    (uintptr_t)output, (uintptr_t)input.data(), BYTES};

int main() {
  uintptr_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  if (!hart) {
    while (!__atomic_load_n(&done, __ATOMIC_ACQUIRE)) {
    }
    return 0;
  }
  if (hart != 1)
    return 0;
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  // The first value reserves its full range; only its initialized row is read.
  // The second value then crosses the six-bit storage-bank index boundary.
  edges[0].source = 0;
  edges[0].target = 4;
  edges[0].source_address = edges[0].target_address = (uintptr_t)reserved;
  edges[0].bytes = HUGE;
  edges[1].source = 1;
  edges[1].target = 2;
  edges[1].source_address = edges[1].target_address = (uintptr_t)temporary;
  edges[1].bytes = BYTES;
  const size_t dependencies[] = {0, 2, 3};
  const unsigned dependentJobs[] = {1, 3, 4};
  for (unsigned i = 0; i < 3; ++i) {
    jobs[dependentJobs[i]].dependencies = &dependencies[i];
    jobs[dependentJobs[i]].dependency_count = 1;
  }
  kernel_result results[5];
  const unsigned threads[] = {0, 2, 3, 3, 0};
  for (unsigned i = 0; i < 5; ++i) {
    unsigned thread = threads[i];
    jobs[i].thread = thread;
    jobs[i].task.id = i + 1;
    jobs[i].task.signature = slots[thread].signature;
    jobs[i].opaque = (void *)&arguments[i];
  }
  if (kernel_run(&owner, jobs, 5, edges, 2, results))
    return 1;
  if (small[0] != prefix[0] || small[1] != prefix[1] || output[0] != input[0] ||
      output[2048] != input[2048] || output[WORDS - 1] != input[WORDS - 1])
    return 2;
  __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
  return 0;
}
