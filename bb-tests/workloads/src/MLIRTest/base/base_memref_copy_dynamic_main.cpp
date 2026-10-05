#include <CRunnerUtils.h>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

using View = StridedMemRefType<int8_t, 1>;
extern "C" void _mlir_ciface_copy_unit(View *, View *);
extern "C" void _mlir_ciface_copy_gap(View *, View *);

static uint64_t cycle() {
  uint64_t value;
  asm volatile("rdcycle %0" : "=r"(value)::"memory");
  return value;
}

int main() {
  constexpr int extent = 192, alignedOffset = 8;
  alignas(64) uint8_t source[extent], destination[extent];
  int cases = 0;
  for (int64_t stride : {int64_t(1), int64_t(2)})
    for (int64_t n :
         {int64_t(0), int64_t(1), int64_t(3), int64_t(16), int64_t(65)})
      for (int64_t offset : {int64_t(0), int64_t(3), int64_t(15)}) {
        for (int i = 0; i < extent; ++i) {
          source[i] = uint8_t(i * 37 + 11);
          destination[i] = 0xa5;
        }
        View input{reinterpret_cast<int8_t *>(source),
                   reinterpret_cast<int8_t *>(source + alignedOffset),
                   offset,
                   {n},
                   {1}};
        View output{reinterpret_cast<int8_t *>(destination),
                    reinterpret_cast<int8_t *>(destination + alignedOffset),
                    offset,
                    {n},
                    {stride}};
        auto copy =
            stride == 1 ? _mlir_ciface_copy_unit : _mlir_ciface_copy_gap;
        const uint64_t start = cycle();
        copy(&input, &output);
        const uint64_t elapsed = cycle() - start;
        for (int i = 0; i < extent; ++i) {
          const int64_t relative = i - alignedOffset - offset;
          const bool copied =
              relative >= 0 && relative % stride == 0 && relative / stride < n;
          const uint8_t expected =
              copied
                  ? uint8_t((alignedOffset + offset + relative / stride) * 37 +
                            11)
                  : 0xa5;
          if (source[i] != uint8_t(i * 37 + 11) || destination[i] != expected) {
            std::printf("FAIL n=%lld offset=%lld stride=%lld byte=%d\n",
                        (long long)n, (long long)offset, (long long)stride, i);
            return 1;
          }
        }
        std::printf("stack_runtime n=%lld offset=%lld stride=%lld cycles=%llu "
                    "guard=PASS\n",
                    (long long)n, (long long)offset, (long long)stride,
                    (unsigned long long)elapsed);
        ++cases;
      }
  std::printf("dynamic copy %d cases full source/destination guard PASS\n",
              cases);
  return cases == 30 ? 0 : 1;
}
