#include <algorithm>
#include <array>
#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>

template <int Rank> struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[Rank], strides[Rank];
};
extern "C" void _mlir_ciface_compute_integer_softmax(tensor<2> *, tensor<1> *,
                                                     tensor<2> *);
constexpr int rows = 8, width = 128, elements = rows * width;
constexpr float score_at(int row, int col) {
  switch (row % 8) {
  case 0:
    return 0;
  case 1:
    return col == (17 + row) % width ? 16 : -16;
  case 2:
    return col > width / 2 ? -3.4028234663852886e38f
                           : float((col + row) % 7 - 3) * .125f;
  case 3:
    return float(col - 64) * (.25f + float(row % 3) * .125f);
  case 4:
  case 5:
    return (float(col % 13 - 6) + (row == 4 ? .5f : -.5f)) / 256;
  case 6:
    return -3.4028234663852886e38f;
  default:
    return float((col + row) % 19 - 9) * 1.25f;
  }
}
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) float input[elements];
    alignas(64) int32_t table[4096];
    alignas(64) float expected[elements + 8];
  } f{};
  double value = 32768;
  for (auto &entry : f.table) {
    entry = int32_t(value + .5);
    value *= .9961013694701175;
  }
  std::fill_n(f.expected, elements + 8, -111.25f);
  for (int row = 0; row < rows; ++row) {
    int32_t fixed[width]{};
    int32_t maximum = -(1 << 20);
    for (int col = 0; col < width; ++col) {
      float score = score_at(row, col);
      f.input[row * width + col] = score;
      fixed[col] =
          __builtin_roundevenf(std::clamp(score, -4096.f, 4096.f) * 256);
      maximum = std::max(maximum, fixed[col]);
    }
    int32_t weights[width]{}, sum = 0;
    for (int col = 0; col < width; ++col) {
      int32_t distance = maximum - fixed[col];
      weights[col] = f.table[distance > 4095 ? 4095 : distance];
      sum += weights[col];
    }
    for (int col = 0; col < width; ++col) {
      int32_t probability = (weights[col] * 32768 + sum / 2) / sum;
      f.expected[4 + row * width + col] = float(probability) / 32768;
    }
  }
  return f;
}();
alignas(64) static constinit auto output = []() constexpr {
  std::array<float, elements + 8> o;
  o.fill(-111.25f);
  return o;
}();
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.data(), fixture.expected, sizeof(output)};
static void readback(void *) {
  bb_mem_alloc(31, 1, 1);
  bb_mvin((uintptr_t)output.data(), 31, sizeof(output) / 16, 1);
  bb_mem_release(31);
}
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  tensor<2> scores{fixture.input, fixture.input, 0, {rows, width}, {width, 1}};
  tensor<1> table{fixture.table, fixture.table, 0, {4096}, {1}};
  tensor<2> result{
      output.data(), output.data() + 4, 0, {rows, width}, {width, 1}};
  _mlir_ciface_compute_integer_softmax(&scores, &table, &result);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[0].signature, 0, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
