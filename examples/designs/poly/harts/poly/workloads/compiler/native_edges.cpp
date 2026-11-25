#include <cstdint>
#include <kernel.h>
struct vector4 {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_edge_after(vector4 *, vector4 *, vector4 *);
extern "C" void _mlir_ciface_edge_transfer(vector4 *, vector4 *);
struct guarded {
  uint32_t before[4], data[4], after[4];
};
alignas(64) static constexpr uint32_t input[2][4] = {{7, 8, 9, 10},
                                                     {101, 102, 103, 104}};
alignas(64) static guarded middle[2] = {
    {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}},
    {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}}};
alignas(64) static guarded output[2][2] = {
    {{{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}},
     {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}}},
    {{{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}},
     {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}}}};
static constexpr guarded expected[2] = {
    {{91, 92, 93, 94}, {7, 8, 9, 10}, {95, 96, 97, 98}},
    {{91, 92, 93, 94}, {101, 102, 103, 104}, {95, 96, 97, 98}}};
static const struct {
  const void *actual, *reference;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {&middle[0], &expected[0], 48},    {&middle[1], &expected[1], 48},
    {&output[0][0], &expected[0], 48}, {&output[0][1], &expected[1], 48},
    {&output[1][0], &expected[0], 48}, {&output[1][1], &expected[1], 48}};
static const struct {
  uint64_t id;
  const void *begin;
  uint64_t end;
} scopes[] __attribute__((section(".bbsim.profile"), used)) = {
    {1, reinterpret_cast<const void *>(kernel_begin), 0},
    {2, reinterpret_cast<const void *>(thread_transfer), 0},
    {3, reinterpret_cast<const void *>(thread_transfer_ready), 0},
    {4, reinterpret_cast<const void *>(thread_transfer_release), 0},
    {5, reinterpret_cast<const void *>(kernel_end), 0}};
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  for (int i = 0; i < 2; ++i) {
    vector4 source{const_cast<uint32_t *>(input[i]),
                   const_cast<uint32_t *>(input[i]),
                   0,
                   {4},
                   {1}};
    vector4 tmp{&middle[i], middle[i].data, 0, {4}, {1}};
    vector4 out{&output[0][i], output[0][i].data, 0, {4}, {1}};
    _mlir_ciface_edge_after(&source, &tmp, &out);
  }
  for (int i = 0; i < 2; ++i) {
    vector4 source{const_cast<uint32_t *>(input[i]),
                   const_cast<uint32_t *>(input[i]),
                   0,
                   {4},
                   {1}};
    vector4 out{&output[1][i], output[1][i].data, 0, {4}, {1}};
    _mlir_ciface_edge_transfer(&source, &out);
  }
  return 0;
}
