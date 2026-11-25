#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define LINK(op, address, data)                                                \
  ({                                                                           \
    uint64_t result;                                                           \
    asm volatile(".insn r 0x2b, 7, " #op ", %0, %1, %2"                        \
                 : "=r"(result)                                                \
                 : "r"((uint64_t)(address)), "r"((uint64_t)(data))             \
                 : "memory");                                                  \
    result;                                                                    \
  })

struct Input {
  uint8_t bytes[8256];
};
alignas(4096) static Input input = []() constexpr {
  Input value{};
  for (unsigned i = 0; i < sizeof(value.bytes); ++i)
    value.bytes[i] = (i * 37 + 11) & 255;
  return value;
}();
struct Outputs {
  uint8_t bytes[20][8256];
};
alignas(4096) Outputs outputs = []() constexpr {
  Outputs value{};
  for (auto &row : value.bytes)
    for (auto &byte : row)
      byte = 0xa5;
  return value;
}();

uint64_t payload_cycles[40];
static const uint64_t timing_reference[40]
    __attribute__((used, section(".bbsim.reference"))) = {};
struct Check {
  uint64_t actual, expected, bytes;
};
static const Check timing_capture
    __attribute__((used, section(".bbsim.capture_output"))) = {
        0, (uintptr_t)timing_reference, sizeof(timing_reference)};

int main(void) {
  const unsigned sizes[] = {1, 7, 8, 15, 16, 17, 65, 4095, 4097, 8192};
  for (unsigned test = 0; test < 2 * sizeof(sizes) / sizeof(sizes[0]); ++test) {
    unsigned item = test % 10, shared = test < 10 ? 64 : 72;
    unsigned bytes = sizes[item],
             offset =
                 item == 6 ? 4095 : (item == 7 || item == 9 ? 0 : item % 16);
    uint8_t *output = outputs.bytes[test];
    // Dirty the destination cache line before DMA, then observe it after
    // completion.
    *(volatile uint8_t *)(output + offset) = 0xa5;
    *(volatile uint8_t *)(input.bytes + offset) = input.bytes[offset];
    uint64_t start, end;
    asm volatile("rdcycle %0" : "=r"(start) : : "memory");
    if (LINK(18, (uintptr_t)(input.bytes + offset), shared) ||
        LINK(19, 1, bytes))
      return 1;
    asm volatile("rdcycle %0" : "=r"(end) : : "memory");
    payload_cycles[2 * test] = end - start;
    /* The scalar endpoint independently observes the first word. */
    uint64_t expected = 0;
    memcpy(&expected, input.bytes + offset, bytes < 8 ? bytes : 8);
    if (LINK(16, shared, 0) != expected)
      return 2;
    asm volatile("rdcycle %0" : "=r"(start) : : "memory");
    if (LINK(18, (uintptr_t)(output + offset), shared) || LINK(19, 0, bytes))
      return 3;
    asm volatile("rdcycle %0" : "=r"(end) : : "memory");
    payload_cycles[2 * test + 1] = end - start;
    if (*(volatile uint8_t *)(output + offset) != input.bytes[offset] ||
        *(volatile uint8_t *)(output + offset + bytes - 1) !=
            input.bytes[offset + bytes - 1])
      return 4;
    for (unsigned i = 0; i < bytes; i += 64)
      if (*(volatile uint8_t *)(output + offset + i) != input.bytes[offset + i])
        return 5;
    if ((offset && output[offset - 1] != 0xa5) ||
        output[offset + bytes] != 0xa5)
      return 6;
  }
  // Fixed raw timing record; no formatting, UART output or DDR snapshot.
  uint64_t hart;
  asm volatile("csrr %0, mhartid" : "=r"(hart));
  volatile uint32_t *capture =
      (volatile uint32_t *)(0x60000000ULL + hart * 0x40000);
  *capture = 0x42424f55;
  for (uint64_t value : payload_cycles) {
    *capture = (uint32_t)value;
    *capture = (uint32_t)(value >> 32);
  }
  return 0;
}
