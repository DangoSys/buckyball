#include <interconnect.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "chip %u line %d FAILED\n", (unsigned)chip, __LINE__);   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main(void) {
  uint64_t chip = link_read(LINK_CHIP_ID);
  uint64_t count = link_read(LINK_CHIP_COUNT);
  CHECK(count > 1 && count <= 32);
  const uint64_t memory = link_read(LINK_MEMORY_BYTES);
  CHECK(memory > (4ULL << 30));
  // Same physical addresses in distinct DDRs, both above a 32-bit address
  // range.
  volatile uint64_t *source = (volatile uint64_t *)(0x80000000ULL + memory / 2);
  volatile uint64_t *result =
      (volatile uint64_t *)(0x80000000ULL + memory * 3 / 4);
  for (unsigned i = 0; i < 32; ++i)
    source[i] = 1000 * chip + i;
  CHECK(chip_send((chip + 1) % count, (uintptr_t)source, (uintptr_t)result, 256,
                  10 + chip) == 0);
  // Thinker ranks publish disjoint partials into the last chip's DDR.
  if (chip < count - 1) {
    while (!link_read(LINK_EVENT_COUNT)) {
    }
    uint64_t previous = (chip + count - 1) % count;
    CHECK(link_read(LINK_EVENT_SOURCE) == previous);
    CHECK(link_read(LINK_EVENT_TAG) == 10 + previous);
    CHECK(link_read(LINK_EVENT_BYTES) == 256);
    __asm__ volatile("fence rw,rw" ::: "memory");
    for (unsigned i = 0; i < 32; ++i)
      CHECK(result[i] == 1000 * previous + i);
    link_write(LINK_EVENT_ACK, 1);
    source[0] = chip + 1;
    CHECK(chip_send(count - 1, (uintptr_t)source,
                    (uintptr_t)(result + 64 + chip), 8, 20 + chip) == 0);
    while (!link_read(LINK_EVENT_COUNT)) {
    }
    CHECK(link_read(LINK_EVENT_SOURCE) == count - 1);
    CHECK(link_read(LINK_EVENT_TAG) == 30);
    CHECK(result[128] == count * (count - 1) / 2);
    link_write(LINK_EVENT_ACK, 1);
  } else {
    unsigned received = 0;
    unsigned ring_seen = 0;
    for (unsigned i = 0; i < count; ++i) {
      while (!link_read(LINK_EVENT_COUNT)) {
      }
      uint64_t peer = link_read(LINK_EVENT_SOURCE);
      if (peer == count - 2 && link_read(LINK_EVENT_TAG) == 10 + count - 2) {
        CHECK(peer == count - 2 && !ring_seen);
        for (unsigned j = 0; j < 32; ++j)
          CHECK(result[j] == 1000 * (count - 2) + j);
        ring_seen = 1;
        link_write(LINK_EVENT_ACK, 1);
        continue;
      }
      CHECK(peer < count - 1 && !(received & (1u << peer)));
      CHECK(link_read(LINK_EVENT_TAG) == 20 + peer);
      received |= 1u << peer;
      link_write(LINK_EVENT_ACK, 1);
    }
    CHECK(ring_seen && received == (1u << (count - 1)) - 1);
    source[0] = 0;
    for (unsigned peer = 0; peer < count - 1; ++peer)
      source[0] += result[64 + peer];
    CHECK(source[0] == count * (count - 1) / 2);
    for (unsigned peer = 0; peer < count - 1; ++peer)
      CHECK(chip_send(peer, (uintptr_t)source, (uintptr_t)(result + 128), 8,
                      30) == 0);
  }
  printf("chip %u: isolated DDR, ring DMA, gather and broadcast PASSED\n",
         (unsigned)chip);
  return 0;
}
