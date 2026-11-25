#include <hartlink.h>
#include <stdint.h>

int main(void) {
  uint64_t capacity = hartlink_query(HARTLINK_CAPACITY);
  if (capacity < 8192)
    return 1;
  const uint64_t positions[] = {0, 16, 4080, 4096, capacity - 16};
  for (unsigned i = 0; i < sizeof(positions) / sizeof(positions[0]); ++i) {
    hartlink_write64(positions[i], UINT64_C(0x0123456789abcdef) ^ i);
    hartlink_write64(positions[i] + 8, UINT64_C(0xfedcba9876543210) ^ i);
  }
  for (unsigned i = 0; i < sizeof(positions) / sizeof(positions[0]); ++i)
    if (hartlink_read64(positions[i]) != (UINT64_C(0x0123456789abcdef) ^ i) ||
        hartlink_read64(positions[i] + 8) != (UINT64_C(0xfedcba9876543210) ^ i))
      return 2;
  return 0;
}
