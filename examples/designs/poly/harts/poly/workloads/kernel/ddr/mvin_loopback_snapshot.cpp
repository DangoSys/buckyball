#include <cstddef>
#include <cstdint>
#include <isa.h>
#include <kernel.h>
struct source_page {
  uint8_t padding[0x4c0], previous[128], current[128];
};
alignas(4096) static constexpr source_page source = {{0},
                                                     {
#embed "mvin_hash_previous.bin"
                                                     },
                                                     {
#embed "mvin_hash_current.bin"
                                                     }};
static_assert(offsetof(source_page, previous) == 0x4c0);
static_assert(offsetof(source_page, current) == 0x540);
struct block {
  uint64_t before[2];
  uint8_t data[128];
  uint64_t after[2];
};
constexpr block empty{{0x1122334455667788, 0x99aabbccddeeff00},
                      {},
                      {0x13579bdf2468ace0, 0xfedcba9876543210}};
constexpr block expected_block() {
  block b = empty;
  for (unsigned i = 0; i < 128; ++i)
    b.data[i] = source.current[i];
  return b;
}
static constexpr block expected[2] = {expected_block(), expected_block()};
alignas(64) static block output[2] = {empty, empty};
static_assert(sizeof(output) == 320);
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.capture_output"), used, retain)) = {
    output, expected, sizeof(output)};
static void exercise(void *) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin(reinterpret_cast<uintptr_t>(source.previous), 1, 8, 1);
  bb_mvin(reinterpret_cast<uintptr_t>(source.current), 1, 8, 1);
  bb_mvout(output[0].data, 1, 8, 1);
  bb_mvin(reinterpret_cast<uintptr_t>(output[0].data), 2, 8, 1);
  bb_mvout(output[1].data, 2, 8, 1);
  bb_mem_release(1);
  bb_mem_release(2);
}
int main() {
  kernel owner;
  kernel_slot slots[thread_query(0, THREAD_FIELD_COUNT)];
  kernel_init(&owner, slots, sizeof(slots) / sizeof(slots[0]));
  stream_begin(1, 0, 0);
  stream_job(slots[0].signature, 0, exercise, nullptr, 0);
  stream_submit();
  stream_wait();
  uintptr_t hart;
  asm volatile("csrr %0,mhartid" : "=r"(hart));
  // Per-hart SCU output ABI, matching the baremetal syscalls.
  const uintptr_t SIM_EXIT_ADDR = 0x60000000ULL + (hart << 18);
  auto *stream = reinterpret_cast<volatile uint32_t *>(SIM_EXIT_ADDR);
  *stream = 0x42424f55;
  auto *words = reinterpret_cast<volatile const uint32_t *>(output);
  for (unsigned i = 0; i < sizeof(output) / 4; ++i)
    *stream = words[i];
  return 0;
}
