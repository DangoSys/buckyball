#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct tensor4 {
  void *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
extern "C" void _mlir_ciface_compute(tensor4 *, tensor4 *);
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) float input[2 * 3 * 5 * 128];
    alignas(64) int8_t expected[1072];
  } value{};
  for (int i = 0; i < 2 * 3 * 5 * 128; ++i)
    value.input[i] = i % 7 - 3;
  for (int b = 0; b < 2; ++b)
    for (int h = 0; h < 2; ++h)
      for (int r = 0; r < 2; ++r) {
        int row = (b * 2 + h) * 2 + r;
        int source = ((b * 3 + h) * 5 + 1 + 2 * r) * 128;
        for (int k = 0; k < 128; ++k)
          value.expected[row * 132 + k] = 32 * ((source + k) % 7 - 3);
        for (int k = 128; k < 132; ++k)
          value.expected[row * 132 + k] = 122;
      }
  return value;
}();
alignas(64) static int8_t output[1072];
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
  tensor4 input{
      fixture.input, fixture.input, 0, {2, 3, 5, 128}, {1920, 640, 128, 1}};
  tensor4 result{output, output, 0, {2, 2, 2, 132}, {528, 264, 132, 1}};
  _mlir_ciface_compute(&input, &result);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
