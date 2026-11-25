#include <array>
#include <cstdint>
#include <kernel.h>
#define BUCKYBALL_STREAM
#include <isa.h>
template <int Rank> struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[Rank], strides[Rank];
};
extern "C" void _mlir_ciface_compute(tensor<2> *, tensor<3> *, tensor<2> *);
constexpr int N = OUTPUT_CHANNELS, K = INPUT_K, PACKED = N + N / 32;
constexpr int A_BYTES = K + K / 32, CHUNK = 512,
              W_BYTES = CHUNK * 5 / 8 + CHUNK / 16;
constexpr int M = INPUT_ROWS;
constexpr int BYTES = (M * PACKED + 47) / 16 * 16;
alignas(64) static constinit auto input = []() constexpr {
  std::array<uint8_t, M * A_BYTES> x{};
  for (int row = 0; row < M; ++row)
    for (int j = 0; j < A_BYTES; ++j)
      x[row * A_BYTES + j] = j < K ? 1 + row % 3 : 127;
  return x;
}();
alignas(64) static constinit auto weights = []() constexpr {
  std::array<uint8_t, 2 * N * W_BYTES> x{};
  for (int chunk = 0; chunk < 2; ++chunk)
    for (int col = 0; col < N; ++col) {
      unsigned code = unsigned(col % 31 - 15) & 31,
               base = (chunk * N + col) * W_BYTES;
      for (int k = 0; k < CHUNK; ++k) {
        unsigned bit = 5 * k, at = base + bit / 8;
        x[at] |= uint8_t(code << (bit % 8));
        if (bit % 8 > 3)
          x[at + 1] |= uint8_t(code >> (8 - bit % 8));
      }
      for (int block = 0; block < CHUNK / 32; ++block) {
        x[base + CHUNK * 5 / 8 + block] = 16 * (chunk + 1);
        x[base + CHUNK * 5 / 8 + CHUNK / 32 + block] = 127 + col / 32;
      }
    }
  return x;
}();
alignas(64) static constinit auto output = []() constexpr {
  std::array<uint8_t, BYTES> x{};
  for (int j = 0; j < BYTES; ++j)
    x[j] = 0x5a;
  return x;
}();
static constexpr auto expected = []() constexpr {
  std::array<uint8_t, BYTES> x{};
  for (int j = 0; j < BYTES; ++j)
    x[j] = 0x5a;
  for (int row = 0; row < M; ++row) {
    int q = 0;
    for (int m = 15 * K * 3 / 2 * (1 + row % 3); m >= 128; m >>= 1)
      ++q;
    for (int j = 0; j < N; ++j) {
      int value = K * 3 / 2 * (1 + row % 3) * (j % 31 - 15);
      x[16 + row * PACKED + j] =
          uint8_t(int(__builtin_roundeven(__builtin_ldexp(double(value), -q))));
    }
    for (int j = 0; j < N / 32; ++j)
      x[16 + row * PACKED + N + j] = 127 + q + j;
  }
  return x;
}();
static const struct {
  const void *actual, *reference;
  uint64_t bytes;
} check __attribute__((section(".gvn.bank_output"), used)) = {
    output.data(), expected.data(), BYTES};
static const struct {
  uint64_t id;
  const void *begin;
  uint64_t end;
} scope __attribute__((section(".bbsim.profile"), used)) = {
    1, reinterpret_cast<const void *>(_mlir_ciface_compute), 0};
static void readback(void *) {
  bb_mem_alloc(15, 1, 1);
  bb_mvin((uintptr_t)output.data(), 15, BYTES / 16, 1);
  bb_mem_release(15);
}
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  tensor<2> a{input.data(), input.data(), 0, {M, A_BYTES}, {A_BYTES, 1}};
  tensor<3> w{weights.data(),
              weights.data(),
              0,
              {2, N, W_BYTES},
              {N * W_BYTES, W_BYTES, 1}};
  tensor<2> o{output.data(), output.data() + 16, 0, {M, PACKED}, {PACKED, 1}};
  _mlir_ciface_compute(&a, &w, &o);
  stream_begin(1, 0, 0);
  stream_job(owner.slots[2].signature, 2, readback, nullptr, 0);
  stream_submit();
  stream_wait();
  return 0;
}
