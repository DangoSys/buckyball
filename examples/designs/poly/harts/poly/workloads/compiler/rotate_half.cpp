#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>
struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[4], strides[4];
};
extern "C" void _mlir_ciface_compute_rotate_half(tensor *, tensor *, tensor *,
                                                 tensor *);
constexpr int rows = 128, elements = 4 * rows * 128;
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) float input[elements], cosine[rows * 128], sine[rows * 128],
        expected[elements + 8];
  } f{};
  for (int i = 0; i < elements; ++i)
    f.input[i] = float(i % 127 - 63) * .25f;
  for (int i = 0; i < rows * 128; ++i) {
    f.cosine[i] = float(i % 7 - 3) * .25f;
    f.sine[i] = float(i % 5 - 2) * .125f;
  }
  for (auto &v : f.expected)
    v = -111.25f;
  for (int head = 0; head < 4; ++head)
    for (int row = 0; row < rows; ++row)
      for (int col = 0; col < 128; ++col) {
        int at = (head * rows + row) * 128 + col;
        float paired = f.input[(head * rows + row) * 128 + (col + 64) % 128];
        if (col < 64)
          paired = -paired;
        float first = f.input[at] * f.cosine[row * 128 + col];
        float second = paired * f.sine[row * 128 + col];
        f.expected[4 + at] = first + second;
      }
  return f;
}();
alignas(64) static constinit auto output = []() constexpr {
  struct {
    float data[elements + 8];
  } o{};
  for (auto &v : o.data)
    v = -111.25f;
  return o;
}();
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.data, fixture.expected, sizeof(output.data)};
static void readback(void *) {
  constexpr int total = sizeof(output.data) / 16;
  for (int row = 0; row < total; row += 1024) {
    int count = total - row < 1024 ? total - row : 1024;
    bb_mem_alloc(31, 1, 1);
    bb_mvin((uintptr_t)output.data + row * 16, 31, count, 1);
    bb_mem_release(31);
  }
}
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  tensor input{fixture.input,
               fixture.input,
               0,
               {1, 4, rows, 128},
               {4 * rows * 128, rows * 128, 128, 1}};
  tensor cosine{fixture.cosine,
                fixture.cosine,
                0,
                {1, 1, rows, 128},
                {rows * 128, rows * 128, 128, 1}};
  tensor sine{fixture.sine,
              fixture.sine,
              0,
              {1, 1, rows, 128},
              {rows * 128, rows * 128, 128, 1}};
  tensor result{output.data,
                output.data + 4,
                0,
                {1, 4, rows, 128},
                {4 * rows * 128, rows * 128, 128, 1}};
  _mlir_ciface_compute_rotate_half(&input, &cosine, &sine, &result);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
