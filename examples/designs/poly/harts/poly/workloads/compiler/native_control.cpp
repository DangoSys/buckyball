#include <cstdint>
#include <cstdlib>
#include <kernel.h>
struct vector0 {
  void *allocated, *aligned;
  int64_t offset, sizes[1], strides[1];
};
extern "C" void _mlir_ciface_compute(vector0 *, vector0 *);
alignas(64) static uint32_t guard[4] = {11, 22, 33, 44};
static constexpr uint32_t expected[4] = {11, 22, 33, 44};
static const struct {
  const void *actual, *reference;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check"), used, retain)) = {
    guard, expected, sizeof(guard)};
static const struct {
  uint64_t id;
  const void *begin;
  uint64_t end;
} scopes[] __attribute__((section(".bbsim.profile"), used)) = {
    {100, reinterpret_cast<const void *>(malloc), 0},
    {101, reinterpret_cast<const void *>(free), 0}};
int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  vector0 output{guard, guard, 0, {0}, {1}};
  vector0 input{guard, guard, 0, {4}, {1}};
  _mlir_ciface_compute(&input, &output);
  return 0;
}
