#include <cstdint>
#include <cstdlib>
#include <kernel.h>
struct vector4 {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute_prep(vector4 *, vector4 *, vector4 *);
extern "C" void _mlir_ciface_compute_after(vector4 *, vector4 *, vector4 *);
struct guarded {
  uint32_t before[4], data[4], after[4];
};
alignas(64) static constexpr uint32_t input[4] = {7, 8, 9, 10};
alignas(64) static guarded middle[2] = {
    {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}},
    {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}}};
alignas(64) static guarded output[2] = {
    {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}},
    {{91, 92, 93, 94}, {55, 55, 55, 55}, {95, 96, 97, 98}}};
static constexpr guarded copy = {
    {91, 92, 93, 94}, {7, 8, 9, 10}, {95, 96, 97, 98}};
static constexpr guarded increment = {
    {91, 92, 93, 94}, {8, 8, 8, 8}, {95, 96, 97, 98}};
static const struct {
  const void *actual, *reference;
  uint64_t bytes;
} checks[] __attribute__((section(".bbsim.check"), used, retain)) = {
    {&middle[0], &copy, sizeof(copy)},
    {&middle[1], &copy, sizeof(copy)},
    {&output[0], &increment, sizeof(copy)},
    {&output[1], &copy, sizeof(copy)}};
static const struct {
  uint64_t id;
  const void *begin;
  uint64_t end;
} scopes[] __attribute__((section(".bbsim.profile"), used)) = {
    {100, reinterpret_cast<const void *>(malloc), 0},
    {101, reinterpret_cast<const void *>(free), 0}};
int main() {
  void *skew = malloc(29);
  __asm__ volatile("" ::"r"(skew) : "memory");
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  vector4 source{const_cast<uint32_t *>(input),
                 const_cast<uint32_t *>(input),
                 0,
                 {4},
                 {1}};
  vector4 a{&middle[0], middle[0].data, 0, {4}, {1}};
  vector4 b{&output[0], output[0].data, 0, {4}, {1}};
  _mlir_ciface_compute_prep(&source, &a, &b);
  vector4 c{&middle[1], middle[1].data, 0, {4}, {1}};
  vector4 d{&output[1], output[1].data, 0, {4}, {1}};
  _mlir_ciface_compute_after(&source, &c, &d);
  free(skew);
  return 0;
}
