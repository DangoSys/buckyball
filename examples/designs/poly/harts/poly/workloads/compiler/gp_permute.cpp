#include <cstdint>
#include <kernel.h>

struct tensor {
  void *allocated, *aligned;
  int64_t offset, sizes[3], strides[3];
};
extern "C" void _mlir_ciface_permute_rows(tensor *, tensor *);
struct Fixture {
  uint32_t input[100], output[112];
};
alignas(64) static Fixture fixture = [] {
  Fixture f{};
  for (unsigned i = 0; i < 96; ++i)
    f.input[2 + i] = 0x7f800000u + i;
  for (auto &value : f.output)
    value = 0xa5a5a5a5;
  return f;
}();

int main() {
  kernel owner;
  size_t count = thread_query(0, THREAD_FIELD_COUNT);
  kernel_slot slots[count];
  kernel_init(&owner, slots, count);
  tensor input{fixture.input, fixture.input, 2, {2, 3, 16}, {48, 16, 1}};
  tensor output{fixture.output, fixture.output, 3, {3, 2, 16}, {32, 16, 1}};
  _mlir_ciface_permute_rows(&input, &output);
  for (unsigned i = 0; i < 112; ++i) {
    if (i < 3 || i >= 99) {
      if (fixture.output[i] != 0xa5a5a5a5)
        return 1;
      continue;
    }
    unsigned index = i - 3, row = index / 32, head = index / 16 % 2;
    if (fixture.output[i] !=
        fixture.input[2 + head * 48 + row * 16 + index % 16])
      return 2;
  }
  return 0;
}
