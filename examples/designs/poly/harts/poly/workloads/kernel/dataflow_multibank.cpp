#include "copy.h"
#include <array>
#include <kernel.h>

enum { WORDS = 4098, BYTES = WORDS * 8 };
alignas(64) static constexpr auto input = [] {
  std::array<uint64_t, WORDS> data{};
  for (unsigned i = 0; i < WORDS; ++i)
    data[i] = UINT64_C(0x1234567800000000) + i * 17;
  return data;
}();
alignas(64) static constexpr auto poison = [] {
  std::array<uint64_t, WORDS> data{};
  for (auto &word : data)
    word = UINT64_C(0xa5a5a5a5a5a5a5a5);
  return data;
}();
alignas(64) static auto source = poison;
alignas(64) static std::array<uint64_t, WORDS> targets[5] = {
    poison, poison, poison, poison, poison};
alignas(64) static uint64_t output[5][WORDS]
    __attribute__((section(".noinit")));
static unsigned done;
static kernel_job jobs[6];
static kernel_transfer edges[5];
static copy_task arguments[6];
__attribute__((section(".bbsim.check"),
               used)) static const uint64_t checks[][3] = {
    {(uintptr_t)output[0], (uintptr_t)input.data(), BYTES},
    {(uintptr_t)output[1], (uintptr_t)input.data(), BYTES},
    {(uintptr_t)output[2], (uintptr_t)input.data(), BYTES},
    {(uintptr_t)output[3], (uintptr_t)input.data(), BYTES},
    {(uintptr_t)output[4], (uintptr_t)input.data(), BYTES},
    {(uintptr_t)source.data(), (uintptr_t)poison.data(), BYTES},
    {(uintptr_t)targets[0].data(), (uintptr_t)poison.data(), BYTES},
    {(uintptr_t)targets[1].data(), (uintptr_t)poison.data(), BYTES},
    {(uintptr_t)targets[2].data(), (uintptr_t)poison.data(), BYTES},
    {(uintptr_t)targets[3].data(), (uintptr_t)poison.data(), BYTES},
    {(uintptr_t)targets[4].data(), (uintptr_t)poison.data(), BYTES}};

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
  arguments[0].input = (uintptr_t)input.data();
  arguments[0].output = (uintptr_t)source.data();
  kernel_result results[6];
  for (unsigned i = 0; i < 5; ++i) {
    arguments[i + 1].input = (uintptr_t)targets[i].data();
    arguments[i + 1].output = (uintptr_t)output[i];
    edges[i].target = i + 1;
    edges[i].source_address = arguments[0].output;
    edges[i].target_address = arguments[i + 1].input;
    edges[i].bytes = BYTES;
  }
  for (unsigned i = 0; i < 6; ++i) {
    unsigned thread = i ? i - 1 : 0;
    arguments[i].rows = BYTES / 16;
    jobs[i].thread = thread;
    jobs[i].task.id = i + 1;
    jobs[i].task.signature = slots[thread].signature;
    jobs[i].opaque = &arguments[i];
  }
  for (unsigned iteration = 0; iteration < 2; ++iteration)
    if (kernel_run(&owner, jobs, 6, edges, 5, results))
      return 1;
  __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
  return 0;
}
