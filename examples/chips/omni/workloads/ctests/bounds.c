#include <interconnect.h>

int main(void) {
  if (link_read(LINK_CHIP_ID) != 0)
    return 0;
  uint64_t bytes = link_read(LINK_MEMORY_BYTES);
  // This descriptor must fail rather than writing past chip 1's DDR.
  chip_send(1, 0x80000000ULL, 0x80000000ULL + bytes - 8, 16, 99);
  return 1;
}
