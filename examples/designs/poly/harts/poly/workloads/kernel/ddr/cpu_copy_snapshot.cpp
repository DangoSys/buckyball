#include <cstddef>
#include <cstdint>
struct source_page {
  uint8_t padding[0x540], current[128];
};
alignas(4096) static constexpr source_page source = {{0},
                                                     {
#embed "mvin_hash_current.bin"
                                                     }};
static_assert(offsetof(source_page, current) == 0x540);
alignas(4096) static source_page writable = {};
static const struct {
  const void *actual, *expected;
  uint64_t bytes;
} check __attribute__((section(".bbsim.check_output"), used, retain)) = {
    writable.current, source.current, 128};
int main() {
  uintptr_t hart;
  asm volatile("csrr %0,mhartid" : "=r"(hart));
  auto *stream =
      reinterpret_cast<volatile uint32_t *>(0x60000000ULL + (hart << 18));
  auto *words = reinterpret_cast<volatile const uint32_t *>(source.current);
  auto *copied = reinterpret_cast<volatile uint32_t *>(writable.current);
  for (unsigned i = 0; i < 32; ++i)
    copied[i] = words[i];
  *stream = 0x42424f55;
  for (unsigned i = 0; i < 32; ++i)
    *stream = copied[i];
  return 0;
}
