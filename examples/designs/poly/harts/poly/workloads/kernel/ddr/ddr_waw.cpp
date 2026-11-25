#include <cstdint>
#include <isa.h>
#include <kernel.h>
enum { Rows = 17, Words = Rows * THREAD_MEM_BANK_WIDTH / 64 };
struct block {
  uint64_t before[2], data[Words], after[2];
};
constexpr block pattern(uint64_t seed) {
  block b{{0x11223344, 0x55667788}, {}, {0x99aabbcc, 0xddeeff00}};
  for (unsigned i = 0; i < Words; ++i)
    b.data[i] = seed + i + 1;
  return b;
}
alignas(64) static constexpr block old_value = pattern(0x1234000000000000);
alignas(64) static constexpr block new_value = pattern(0x5678000000000000);
alignas(64) static block output = pattern(0);
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    &output, &new_value, sizeof(block)};
static void exercise(void *) {
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin(reinterpret_cast<uintptr_t>(new_value.data), 2, Rows, 1);
  bb_mset_clear(1, 1, 1);
  bb_mvin_strided(reinterpret_cast<uintptr_t>(old_value.data), 1, 0, 0, 1, Rows,
                  THREAD_MEM_BANK_WIDTH / 8, THREAD_MEM_BANK_WIDTH / 8, 0);
  bb_mvout(output.data, 1, Rows, 1);
  bb_mvout(output.data, 2, Rows, 1);
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
  return 0;
}
