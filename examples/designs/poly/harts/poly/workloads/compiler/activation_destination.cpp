#include <algorithm>
#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>

template <int Rank> struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[Rank], strides[Rank];
};
extern "C" void _mlir_ciface_compute_activation_destination(tensor<2> *,
                                                            tensor<1> *,
                                                            tensor<2> *);
constexpr int64_t round_shift(int64_t value, int shift) {
  return __builtin_roundeven(__builtin_ldexp(double(value), -shift));
}
constexpr int64_t lookup(int code, int exponent, const int32_t *table) {
  if (exponent < -24)
    return int64_t(code) * 32768;
  int64_t x = int64_t(code) * (1LL << ((exponent > 4 ? 4 : exponent) + 26));
  if (x >= (16LL << 26))
    return int64_t(code) * 65536;
  if (x <= -(16LL << 26))
    return 0;
  uint64_t at = x + (16LL << 26), i = at >> 19;
  int64_t a = table[i], b = i == 4095 ? 16LL << 26 : table[i + 1];
  return round_shift(a + round_shift((b - a) * (at & ((1 << 19) - 1)), 19),
                     exponent + 10);
}
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) int32_t table[4096];
    uint8_t input[135], expected[160];
  } f{};
  for (int i = 0; i < 4096; ++i) {
    double x = double(i) / 128 - 16;
    double y =
        x * .5 * (1 + __builtin_erf(x * .7071067811865475244)) * (1 << 26);
    f.table[i] = __builtin_roundeven(y);
  }
  for (auto &byte : f.expected)
    byte = 0x5a;
  const int exponents[] = {-3, 0, 5, -40};
  for (int block = 0; block < 4; ++block) {
    int64_t values[32]{};
    uint64_t maximum = 0;
    for (int i = 0; i < 32; ++i) {
      int code = (i * 13 + block * 3) % 127 - 63;
      f.input[3 + block * 32 + i] = int8_t(code);
      values[i] = lookup(code, exponents[block], f.table);
      maximum = std::max(maximum, uint64_t(__builtin_llabs(values[i])));
    }
    f.input[3 + 128 + block] = exponents[block] + 127;
    int exponent =
        maximum ? exponents[block] + 41 - __builtin_clzll(maximum) : 0;
    exponent = exponent < -127 ? -127 : exponent > 127 ? 127 : exponent;
    for (int i = 0; i < 32; ++i) {
      int64_t code = round_shift(values[i], exponent - exponents[block] + 16);
      f.expected[5 + block * 32 + i] = int8_t(code < -127  ? -127
                                              : code > 127 ? 127
                                                           : code);
    }
    f.expected[5 + 128 + block] = exponent + 127;
  }
  return f;
}();
alignas(64) static constinit auto output = []() constexpr {
  struct {
    uint8_t data[160];
  } result{};
  for (auto &byte : result.data)
    byte = 0x5a;
  return result;
}();
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.data, fixture.expected, 160};
static void readback(void *) {
  bb_mem_alloc(31, 1, 1);
  bb_mvin((uintptr_t)output.data, 31, 10, 1);
  bb_mem_release(31);
}
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  tensor<2> input{fixture.input, fixture.input, 3, {1, 132}, {132, 1}};
  tensor<1> table{fixture.table, fixture.table, 0, {4096}, {1}};
  tensor<2> out{output.data, output.data, 5, {1, 132}, {132, 1}};
  _mlir_ciface_compute_activation_destination(&input, &table, &out);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[2].signature, 2, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
