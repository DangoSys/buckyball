#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct matrix {
  void *allocated, *aligned;
  int64_t offset, sizes[2], strides[2];
};
struct cube {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_compute(matrix *, cube *, matrix *);
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) float input[4 * 512];
    alignas(64) int8_t weights[16 * 528];
    float expected[4 * 16];
  } value{};
  for (int m = 0; m < 4; ++m)
    for (int k = 0; k < 512; ++k)
      value.input[m * 512 + k] = (m * 3 + k) % 7 - 3;
  for (int n = 0; n < 16; ++n)
    for (int k = 0; k < 528; ++k)
      value.weights[n * 528 + k] = k < 512 ? (n + k * 2) % 5 - 2 : 127;
  for (int m = 0; m < 4; ++m)
    for (int n = 0; n < 16; ++n)
      for (int k = 0; k < 512; ++k)
        value.expected[m * 16 + n] +=
            4 * value.input[m * 512 + k] * value.weights[n * 528 + k];
  return value;
}();
alignas(64) static float output[4 * 16] = {17};
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
  matrix input{fixture.input, fixture.input, 0, {4, 512}, {512, 1}};
  cube weights{
      fixture.weights, fixture.weights, 0, {1, 16, 528}, {16 * 528, 528, 1}};
  matrix result{output, output, 0, {4, 16}, {16, 1}};
  _mlir_ciface_compute(&input, &weights, &result);
  const unsigned markers[] = {0, 16, 32, 63};
  for (unsigned i : markers)
    if (output[i] != fixture.expected[i])
      return 1;
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
