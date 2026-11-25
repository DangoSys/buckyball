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
struct sparse {
  uint64_t rows[THREAD_MEM_BANK_ENTRIES][4096 / 8];
};
constexpr sparse sparse_pattern(uint64_t seed) {
  sparse result{};
  for (unsigned row = 0; row < THREAD_MEM_BANK_ENTRIES; ++row)
    for (unsigned col = 0; col < THREAD_MEM_BANK_WIDTH / 64; ++col)
      result.rows[row][col] =
          seed + row * (THREAD_MEM_BANK_WIDTH / 64) + col + 1;
  return result;
}
alignas(4096) static constexpr sparse input0 =
    sparse_pattern(0x1234000000000000);
static constexpr block expected0 = pattern(0x1234000000000000);
alignas(64) static constexpr block input1 = pattern(0x5678000000000000);
alignas(64) static block temporary0 = pattern(0), temporary1 = pattern(0);
alignas(64) static block output0 = pattern(0), output1 = pattern(0);
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {&temporary0, &expected0, sizeof(block)},
    {&output0, &expected0, sizeof(block)},
    {&temporary1, &input1, sizeof(block)},
    {&output1, &input1, sizeof(block)}};
struct arguments {
  const void *input;
  block *temporary, *output;
  bool sparse;
};
static void exercise(void *opaque) {
  auto *a = static_cast<arguments *>(opaque);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  if (a->sparse)
    bb_mvin_blocks(reinterpret_cast<uintptr_t>(a->input), 1, 0, 0, 1,
                   THREAD_MEM_BANK_ENTRIES, 4096 / (THREAD_MEM_BANK_WIDTH / 8),
                   0);
  else
    bb_mvin(reinterpret_cast<uintptr_t>(a->input), 1, Rows, 1);
  bb_mvout(a->temporary->data, 1, Rows, 1);
  bb_mvin(reinterpret_cast<uintptr_t>(a->temporary->data), 2, Rows, 1);
  bb_mvout(a->output->data, 2, Rows, 1);
  bb_mem_release(1);
  bb_mem_release(2);
}
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  arguments a0{input0.rows, &temporary0, &output0, true};
  arguments a1{input1.data, &temporary1, &output1, false};
  stream_begin(2, 0, 0);
  stream_job(slots[0].signature, 0, exercise, &a0, 0);
  stream_job(slots[1].signature, 1, exercise, &a1, 0);
  stream_submit();
  stream_wait();
  return 0;
}
