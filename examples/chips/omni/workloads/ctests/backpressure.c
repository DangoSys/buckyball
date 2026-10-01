#include <interconnect.h>
#include <stdio.h>

int main(void) {
  uint64_t chip = link_read(LINK_CHIP_ID);
  uint64_t memory = link_read(LINK_MEMORY_BYTES);
  volatile uint64_t *source = (volatile uint64_t *)(0x80000000ULL + memory / 2);
  volatile uint64_t *target =
      (volatile uint64_t *)(0x80000000ULL + memory * 3 / 4);
  uint64_t capacity = link_read(LINK_EVENT_CAPACITY);
  if (chip > 1)
    return 0;
  if (chip == 0) {
    for (uint64_t index = 0; index <= capacity; ++index) {
      *source = 100 + index;
      if (chip_send(1, (uintptr_t)source, (uintptr_t)(target + index), 8,
                    index))
        return 1;
    }
  } else {
    // Fill the completion queue before allowing any consumption.
    while (link_read(LINK_EVENT_COUNT) < capacity) {
    }
    if (link_read(LINK_EVENT_COUNT) != capacity)
      return 2;
    for (uint64_t index = 0; index <= capacity; ++index) {
      while (!link_read(LINK_EVENT_COUNT)) {
      }
      if (link_read(LINK_EVENT_SOURCE) != 0 ||
          link_read(LINK_EVENT_TAG) != index || target[index] != 100 + index)
        return 3;
      link_write(LINK_EVENT_ACK, 1);
    }
    puts("completion queue backpressure and ordered delivery PASSED");
  }
  return 0;
}
