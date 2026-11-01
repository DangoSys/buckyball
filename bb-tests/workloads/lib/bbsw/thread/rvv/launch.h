// Runs one RVV kernel image on the GP domain of the calling core's thread.
//
// Banks 0-3 are allocated with one group each; 2 and 3 become the read bank,
// 0 and 1 the write bank. The kernel sees the groups as addresses:
//   0 << 32, 1 << 32  read groups 0, 1   (inputs)
//   2 << 32           write group 0      (the launch descriptor)
//   3 << 32           write group 1      (the output)
// An image starts with the header of kernel.ld: word 1 = code bytes,
// word 2 = entry.
#pragma once
#include <bbhw/isa/isa.h>
#include <stdint.h>

#define RVV_READ_BANK 4
#define RVV_WRITE_BANK 5
#define RVV_PROGRAM_BANK CORE_VIRTUAL_BANK_COUNT
#define RVV_STACK 0x40002000
#define RVV_DESCRIPTOR (UINT64_C(2) << 32)
#define RVV_FLAGS (RVV_DESCRIPTOR + 88)

static inline uint32_t rvv_image_word(const uint8_t *image, int index) {
  return image[4 * index] | image[4 * index + 1] << 8 |
         image[4 * index + 2] << 16 | (uint32_t)image[4 * index + 3] << 24;
}

static inline void rvv_load(const uint8_t *image, uint32_t bytes) {
  for (unsigned bank = 0; bank < 4; ++bank)
    bb_mem_alloc(bank, 1, 1);
  bb_mem_transfer(2, RVV_READ_BANK);
  bb_mem_transfer(3, RVV_READ_BANK);
  bb_mem_transfer(0, RVV_WRITE_BANK);
  bb_mem_transfer(1, RVV_WRITE_BANK);
  mvin_kernel(image, bytes, RVV_PROGRAM_BANK);
}

// Moves the descriptor (a kernel_launch, then any extra data; a multiple of
// 16 bytes) into write group 0 and runs the kernel.
static inline void rvv_run(const void *descriptor, uint32_t bytes) {
  bb_mvin_group((uintptr_t)descriptor, RVV_WRITE_BANK, 0, bytes / 16, 1);
  run_kernel(RVV_READ_BANK, RVV_PROGRAM_BANK, RVV_WRITE_BANK, 0);
}

// The FP exception flags a kernel stored at RVV_FLAGS (the reserved field).
static inline uint64_t rvv_flags(void) {
  struct kernel_launch descriptor __attribute__((aligned(16)));
  bb_mvout_group((uintptr_t)&descriptor, RVV_WRITE_BANK, 0,
                 sizeof(descriptor) / 16, 1);
  return descriptor.reserved;
}

static inline void rvv_release(void) {
  bb_mem_release(RVV_READ_BANK);
  bb_mem_release(RVV_WRITE_BANK);
  release_kernel(RVV_PROGRAM_BANK);
}
