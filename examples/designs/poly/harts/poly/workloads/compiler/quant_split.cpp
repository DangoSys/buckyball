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
    alignas(64) int8_t input[4 * 33];
    alignas(64) int8_t weights[640 * 33];
    alignas(64) int8_t expected[1328];
  } value{};
  for (int m = 0; m < 4; ++m) {
    for (int k = 0; k < 33; ++k)
      value.input[m * 33 + k] = k < 32 ? 64 : 121;
    for (int k = 0; k < 320; ++k)
      value.expected[m * 330 + k] = 32 * ((k + 320) % 7 - 3);
    for (int k = 320; k < 330; ++k)
      value.expected[m * 330 + k] = 127;
  }
  for (int n = 0; n < 640; ++n)
    for (int k = 0; k < 33; ++k)
      value.weights[n * 33 + k] = k < 32 ? 32 * (n % 7 - 3) : 122;
  return value;
}();
alignas(64) static int8_t output[1328];
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
  matrix input{fixture.input, fixture.input, 0, {4, 33}, {33, 1}};
  cube weights{
      fixture.weights, fixture.weights, 0, {1, 640, 33}, {640 * 33, 33, 1}};
  matrix result{output, output, 0, {4, 330}, {330, 1}};
  _mlir_ciface_compute(&input, &weights, &result);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
