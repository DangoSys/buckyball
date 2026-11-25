#include "buckyball.h"
#include <bbhw/isa/isa.h>
#include <rvv/launch.h>
#include <stdio.h>
#include <thread_run.h>

#define ELEMENTS 256

/* The operator of rvv_integer_test over 64 bank rows (256 elements), its
 * output moved out as soon as the operator completes: the move out must see
 * every row the operator wrote, the last ones included. In SmolVLA a GP
 * operator writes 64 rows that a move out takes at once.
 * out = x & 1 ? ((x & 15) << 3) + 1000 : x; 100000 if x & 2; 9 if x & 4. */
static const uint32_t image[] __attribute__((aligned(16))) = {
    0x31564b52, 0x70, 0, 0x40000000, 0, 0, /* header */
    0x06060663, /* 00: beqz a2, 6c                     */
    0x00000693, /* 04: li a3, 0                        */
    0x3e800713, /* 08: li a4, 1000                     */
    0x000187b7, /* 0c: lui a5, 0x18                    */
    0x6a078793, /* 10: addi a5, a5, 1696  (100000)     */
    0x40d60833, /* 14: sub a6, a2, a3                  */
    0x00269893, /* 18: slli a7, a3, 2                  */
    0x05087857, /* 1c: vsetvli a6, a6, e32, m1, ta, mu */
    0x011582b3, /* 20: add t0, a1, a7                  */
    0x0202e507, /* 24: vle32.v v10, (t0)               */
    0x011508b3, /* 28: add a7, a0, a7                  */
    0x26a7b457, /* 2c: vand.vi v8, v10, 15             */
    0x26a0b4d7, /* 30: vand.vi v9, v10, 1              */
    0x26a135d7, /* 34: vand.vi v11, v10, 2             */
    0x26a23657, /* 38: vand.vi v12, v10, 4             */
    0x9681b6d7, /* 3c: vsll.vi v13, v8, 3              */
    0x66903057, /* 40: vmsne.vi v0, v9, 0              */
    0x66b03457, /* 44: vmsne.vi v8, v11, 0             */
    0x66c034d7, /* 48: vmsne.vi v9, v12, 0             */
    0x00d74557, /* 4c: vadd.vx v10, v13, a4, v0.t      */
    0x5e040057, /* 50: vmv.v.v v0, v8                  */
    0x5ca7c457, /* 54: vmerge.vxm v8, v10, a5, v0      */
    0x5e048057, /* 58: vmv.v.v v0, v9                  */
    0x5c84b457, /* 5c: vmerge.vim v8, v8, 9, v0        */
    0x00d806b3, /* 60: add a3, a6, a3                  */
    0x0208e427, /* 64: vse32.v v8, (a7)                */
    0xfac6e6e3, /* 68: bltu a3, a2, 14                 */
    0x00008067  /* 6c: ret                             */
};
static const struct operator_launch call __attribute__((aligned(16))) = {
    0, 0x70, RVV_STACK, {UINT64_C(3) << 32, 0, ELEMENTS}, 0};
static uint32_t input[ELEMENTS] __attribute__((aligned(64)));
static uint32_t output[ELEMENTS] __attribute__((aligned(64)));

/* The Ball instructions, on the hart's thread. */
static void run(void *unused) {
  (void)unused;
  rvv_load((const uint8_t *)image, sizeof(image));
  bb_mvin_group((uintptr_t)input, RVV_READ_BANK, 0, ELEMENTS / 4, 1);
  rvv_run(&call, sizeof(call));
  bb_mvout_group((uintptr_t)output, RVV_WRITE_BANK, 1, ELEMENTS / 4, 1);
  rvv_release();
}

int main(void) {
  for (unsigned i = 0; i < ELEMENTS; ++i) {
    input[i] = (i * 2654435761u) >> 7;
    output[i] = 0xdeadbeef;
  }
  bb_thread_run(run, 0);
  for (unsigned i = 0; i < ELEMENTS; ++i) {
    uint32_t x = input[i];
    uint32_t expected = x & 1 ? ((x & 15) << 3) + 1000 : x;
    if (x & 2)
      expected = 100000;
    if (x & 4)
      expected = 9;
    if (output[i] != expected) {
      printf("RVV rows: element %u is %u, not %u\n", i, (unsigned)output[i],
             (unsigned)expected);
      return 1;
    }
  }
  printf("RVV rows PASSED\n");
  return 0;
}
