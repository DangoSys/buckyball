#include <cstdint>
#include <cstdio>
#include <cstdlib>

static void fail() {
#ifdef BAREMETAL
  volatile uint32_t *sim_exit =
      reinterpret_cast<volatile uint32_t *>(0x60000000);
  *sim_exit = 1;
  while (true) {
  }
#else
  std::exit(1);
#endif
}

extern "C" void check_result(int32_t *allocated, int32_t *aligned,
                             int64_t offset, int64_t size0, int64_t size1,
                             int64_t stride0, int64_t stride1) {
  (void)allocated;
  if (size0 != 1024 || size1 != 8 || stride0 != 8 || stride1 != 1) {
    std::printf("FAILED: matadd bank shape %ldx%ld\n", size0, size1);
    fail();
  }
  int32_t *c = aligned + offset;
  for (int64_t line = 0; line < size0; ++line) {
    for (int64_t column = 0; column < size1; ++column) {
      size_t index = static_cast<size_t>(line * stride0 + column * stride1);
      uint32_t expected =
          static_cast<uint32_t>(line) + static_cast<uint32_t>(column);
      if (static_cast<uint32_t>(c[index]) != expected) {
        std::printf("FAILED: matadd bank[%ld][%ld] expected %d got %d\n", line,
                    column, static_cast<int32_t>(expected), c[index]);
        fail();
      }
    }
  }
  std::printf("PASSED: matadd bank 1024x8\n");
}
