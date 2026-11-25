#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>

template <int Rank> struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[Rank], strides[Rank];
};
extern "C" void _mlir_ciface_compute_columns(tensor<2> *, tensor<3> *,
                                             tensor<2> *);
struct outputs {
  float single[160 + 8], many[5 * 160 + 8];
};
constexpr int CHUNK = COLUMN_K, CHUNKS = COLUMN_CHUNKS, K = CHUNK * CHUNKS;
constexpr int INPUT = K + K / 32, WEIGHT = CHUNK + CHUNK / 32;
static constexpr outputs blank = []() constexpr {
  outputs value{};
  for (float &x : value.single)
    x = 17;
  for (float &x : value.many)
    x = 17;
  return value;
}();
// Build the fixtures and integer reference at compile time, outside simulation.
static constinit auto fixture = []() constexpr {
  struct {
    alignas(64) int8_t input[5 * INPUT + 8];
    alignas(64) int8_t weights[CHUNKS * 160 * WEIGHT + 8];
    outputs expected;
  } result{};
  for (int m = 0; m < 5; ++m)
    for (int k = 0; k < INPUT; ++k)
      result.input[3 + m * INPUT + k] = k < K ? (m * 5 + k) % 7 - 3 : 127;
  for (int part = 0; part < CHUNKS; ++part)
    for (int n = 0; n < 160; ++n)
      for (int k = 0; k < WEIGHT; ++k)
        result.weights[5 + (part * 160 + n) * WEIGHT + k] =
            k < CHUNK ? (n * 3 + (part * CHUNK + k) * 7) % 17 - 8 : 127;
  result.expected = blank;
  for (int m = 0; m < 5; ++m)
    for (int n = 0; n < 17; ++n) {
      int expected = 0;
      for (int k = 0; k < K; ++k)
        expected += ((m * 5 + k) % 7 - 3) * ((n * 3 + k * 7) % 17 - 8);
      for (int column = n; column < 160; column += 17) {
        result.expected.many[4 + m * 160 + column] = expected;
        if (!m)
          result.expected.single[4 + column] = expected;
      }
    }
  return result;
}();
alignas(64) static constinit outputs output = blank;
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    COLUMN_ROWS == 1 ? output.single : output.many,
    COLUMN_ROWS == 1 ? fixture.expected.single : fixture.expected.many,
    (COLUMN_ROWS * 160 + 8) * sizeof(float)};
static void readback(void *) {
  bb_mem_alloc(31, 1, 1);
  bb_mvin((uintptr_t)check.actual, 31, check.bytes / 16, 1);
  bb_mem_release(31);
}

int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  // Keep dirty CPU data at both ends of the unaligned DMA inputs.
  *(volatile int8_t *)&fixture.input[3] = -3;
  *(volatile int8_t *)&fixture.input[3 + INPUT - 1] = 127;
  *(volatile int8_t *)&fixture.weights[5] = -8;
  *(volatile int8_t *)&fixture.weights[5 + CHUNKS * 160 * WEIGHT - 1] = 127;
  tensor<2> a{
      fixture.input, fixture.input, 3, {COLUMN_ROWS, INPUT}, {INPUT, 1}};
  tensor<3> w{fixture.weights,
              fixture.weights,
              5,
              {CHUNKS, 160, WEIGHT},
              {160 * WEIGHT, WEIGHT, 1}};
  const unsigned columns[] = {0, 31, 32, 159};
  float *actual = COLUMN_ROWS == 1 ? output.single : output.many;
  const float *expected =
      COLUMN_ROWS == 1 ? fixture.expected.single : fixture.expected.many;
  tensor<2> out{actual, actual, 4, {COLUMN_ROWS, 160}, {160, 1}};
  _mlir_ciface_compute_columns(&a, &w, &out);
  for (unsigned n : columns)
    if (actual[4 + (COLUMN_ROWS - 1) * 160 + n] !=
        expected[4 + (COLUMN_ROWS - 1) * 160 + n])
      return 1;
  stream_begin(1, 0, 0);
  stream_job(owner.slots[2].signature, 2, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
