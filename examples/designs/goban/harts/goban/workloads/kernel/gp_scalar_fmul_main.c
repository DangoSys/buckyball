#include <bbhw/isa/isa.h>
#include <stdint.h>
#include <stdio.h>
#include <thread.h>

static const unsigned char image[] __attribute__((aligned(8))) = {
#embed "gp_scalar_fmul.bin"
};
static const uint32_t operator_image[] __attribute__((aligned(16))) = {
    0x31564b52,
    0x9f8,
    0x9e0,
    0x40000000,
    0,
    0,
    [6 + 0x9e0 / 4] = 0xf00587d3, // fmv.w.x fa5, a1
    0xf0060753,                   // fmv.w.x fa4, a2
    0x10f7f7d3,                   // fmul.s fa5, fa5, fa5, dyn
    0xe00785d3,                   // fmv.x.w a1, fa5
    0x00b53023,                   // sd a1, 0(a0)
    0x00008067                    // ret
};
static const struct operator_launch call __attribute__((aligned(16))) = {
    0x9e0,
    0x9f8,
    0x40002000,
    {UINT64_C(3) << 32, UINT64_C(0xffffffffc003e150),
     UINT64_C(0xffffffffc003e150), 0, 0, 0, 0, 0},
    0};
static uint64_t output[2] __attribute__((aligned(64)));
int main(void) {
  uint32_t thread = 0;
  uint64_t signature = thread_query(thread, THREAD_FIELD_SIGNATURE);
  uint64_t base = thread_query(thread, THREAD_FIELD_TLS_BASE);
  thread_tss_write(0, image, sizeof(image));
  thread_load(thread, THREAD_CODE, 0, 0, (sizeof(image) + 7) & ~(size_t)7);
  uint64_t args[] = {(uintptr_t)operator_image, sizeof(operator_image),
                     (uintptr_t)&call, (uintptr_t)output};
  struct thread_task task = {1,
                             0,
                             sizeof(image),
                             base,
                             base +
                                 thread_query(thread, THREAD_FIELD_TLS_BYTES),
                             signature};
  printf("GP FMUL START thread=%u entry=9e0 target_pc=9e8\n", thread);
  thread_acquire();
  thread_start_data(thread, &task, args, sizeof(args));
  int cancelled;
  uint64_t result = thread_wait(thread, task.id, &cancelled);
  thread_release();
  printf("GP FMUL DONE value=%016lx guard=%016lx result=%lx cancelled=%d\n",
         (unsigned long)output[0], (unsigned long)output[1],
         (unsigned long)result, cancelled);
  if (result || cancelled || output[0] != UINT64_C(0x4087e0bc) || output[1])
    return 1;
  printf("GP FMUL PASSED\n");
  return 0;
}
