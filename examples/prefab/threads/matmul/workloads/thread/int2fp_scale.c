// int2fp on the thread alone: 16 INT32 lanes times their FP32 scales. The
// thread has no FPU: the scales and the expected results are FP32 bit
// patterns.
#include <bbhw/isa/isa.h>
#include <isa/int2fp.h>
#include <stdint.h>

static const uint32_t scale[16] = {
    0x3e000000, 0x3e800000, 0x3ec00000, 0x3f000000, 0x3f200000, 0x3f400000,
    0x3f600000, 0x3f800000, 0x3d800000, 0x3e000000, 0x3e400000, 0x3e800000,
    0x3f900000, 0x3fa00000, 0x3fb00000, 0x3fc00000};
static const uint32_t expected[16] = {
    0x3f800000, 0x40000000, 0x40400000, 0x40800000, 0x40a00000, 0x40c00000,
    0x40e00000, 0x41000000, 0x3f000000, 0x3f800000, 0x3fc00000, 0x40000000,
    0x41100000, 0x41200000, 0x41300000, 0x41400000};
#define INPUT ((volatile int32_t *)HOST_TSS_BASE)
#define SCALE ((volatile uint32_t *)(HOST_TSS_BASE + 64))
#define OUTPUT ((volatile uint32_t *)(HOST_TSS_BASE + 128))

uint64_t entry(void) {
  for (int i = 0; i < 16; ++i) {
    INPUT[i] = 8;
    SCALE[i] = scale[i];
    OUTPUT[i] = 0;
  }
  bb_mem_alloc(0, 4, 1);
  bb_mem_alloc(1, 4, 1);
  bb_mem_alloc(2, 4, 1);
  bb_mvin((uintptr_t)INPUT, 0, 4, 1);
  bb_mvin((uintptr_t)SCALE, 1, 4, 1);
  bb_int32_to_fp32(0, 1, 2, 4, 0);
  bb_mvout((uintptr_t)OUTPUT, 2, 4, 1);
  asm volatile("fence" ::: "memory");
  for (int i = 0; i < 16; ++i)
    if (OUTPUT[i] != expected[i])
      return 1 + i;
  bb_mem_release(0);
  bb_mem_release(1);
  bb_mem_release(2);
  return 0;
}
