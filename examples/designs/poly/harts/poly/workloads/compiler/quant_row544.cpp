#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
extern "C" void _mlir_ciface_compute(matrix *, matrix *);
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) float input[4 * 544];
    alignas(64) int8_t expected[2256];
  } value{};
  for (int m = 0; m < 4; ++m) {
    for (int k = 0; k < 544; ++k) {
      value.input[m * 544 + k] = (m * 3 + k) % 7 - 3;
      value.expected[m * 561 + k] = 32 * ((m * 3 + k) % 7 - 3);
    }
    for (int k = 544; k < 561; ++k)
      value.expected[m * 561 + k] = 123;
  }
  return value;
}();
alignas(64) static int8_t output[2256];
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output, fixture.expected, sizeof(output)};
static void readback(void *) {
  bb_mem_alloc(31, 1, 1);
  bb_mvin((uintptr_t)output, 31, sizeof(output) / 16, 1);
  bb_mem_release(31);
}
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  matrix input{fixture.input, fixture.input, 0, {4, 544}, {544, 1}};
  matrix result{output, output, 0, {4, 561}, {561, 1}};
  _mlir_ciface_compute(&input, &result);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
