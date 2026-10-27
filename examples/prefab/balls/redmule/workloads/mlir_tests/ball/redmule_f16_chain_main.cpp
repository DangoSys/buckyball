#include <bbhw/isa/isa.h>
#include <cstdio>

extern "C" void _mlir_ciface_redmule_chain();
alignas(64) static uint16_t a[16], b[64], c[16];

int main() {
  for (auto &value : a)
    value = 0x3c00;
  for (auto &value : b)
    value = 0x3800;
  bb_mem_alloc(1, 1, 1);
  bb_mem_alloc(2, 1, 1);
  bb_mem_alloc(3, 1, 1);
  bb_mvin((uintptr_t)a, 1, 2, 1);
  bb_mvin((uintptr_t)b, 2, 8, 1);
  _mlir_ciface_redmule_chain();
  bb_mvout((uintptr_t)c, 3, 2, 1);
  for (auto value : c)
    if (value != 0x4800)
      __builtin_trap();
  bb_mem_release(1);
  bb_mem_release(2);
  bb_mem_release(3);
  puts("RedMulE BallOp FP16 chain PASS");
  return 0;
}
