// toint8 on the thread alone: INT32 to INT8 with unit scales and ReLU, four
// pixels into rows 3, 4, 7 and 8 of the output bank, checked by the thread.
#include <bbhw/isa/isa.h>
#include <isa/quant.h>
#include <stdint.h>

#define INPUT ((volatile int32_t *)HOST_TSS_BASE)
#define SCALES ((volatile uint32_t *)(HOST_TSS_BASE + 256))
#define OUTPUT ((volatile int8_t *)(HOST_TSS_BASE + 320))

uint64_t entry(void) {
  for (int channel = 0; channel < 16; ++channel)
    SCALES[channel] = 0x3f800000; // 1.0f
  for (int position = 0; position < 4; ++position)
    for (int channel = 0; channel < 16; ++channel)
      INPUT[position * 16 + channel] = position * 20 + channel - 8;
  for (int i = 0; i < 192; ++i)
    OUTPUT[i] = 0;
  bb_mem_alloc(0, 1, 1);
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mvin((uintptr_t)INPUT, 0, 16, 1);
  bb_mvin((uintptr_t)SCALES, 1, 4, 1);
  bb_quant_i32_to_i8(0, 1, 2, 16, 0, 3, 2, 2, 4, 1);
  bb_mvout((uintptr_t)OUTPUT, 2, 12, 1);
  asm volatile("fence" ::: "memory");
  for (int row = 0; row < 4; ++row)
    for (int channel = 0; channel < 16; ++channel) {
      int32_t value = INPUT[row * 16 + channel];
      int output_row = 3 + (row / 2) * 4 + row % 2;
      if (OUTPUT[output_row * 16 + channel] != (value < 0 ? 0 : (int8_t)value))
        return 1 + row * 16 + channel;
    }
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
