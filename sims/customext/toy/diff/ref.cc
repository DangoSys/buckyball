
#include "toy.h"
#include <cstdint>
#include <cstdio>

// void difftest_memcpy(paddr_t addr, void *buf, size_t n, bool direction) {
//   if (direction == DIFFTEST_TO_DUT) {
//     buf = (void *)guest_to_host(addr);
//   }
//   else if (direction == DIFFTEST_TO_REF) {
//     for (size_t i = 0; i < n; i++) {
//       paddr_write(addr + i, 1, *((uint8_t*)buf + i));
//     }
//   }
// }

// 在DUT host memory的`buf`和REF guest memory的`dest`之间拷贝`n`字节,
// `direction`指定拷贝的方向, `DIFFTEST_TO_DUT`表示往DUT拷贝,
// `DIFFTEST_TO_REF`表示往REF拷贝

// void difftest_regcpy(void *dut, bool direction) {
//   if (direction == DIFFTEST_TO_DUT) {
//     for (int i = 0; i < 32; i++) { ((CPU_state *)dut)->gpr[i] = cpu.gpr[i]; }
//     ((CPU_state *)dut)->pc = cpu.pc;
//     for (int i = 0; i <  4; i++) { ((CPU_state *)dut)->csr[i] = cpu.csr[i]; }
//     // printf("%lx  %lx\n", ((CPU_state *)dut)->pc, cpu.pc);
//   }
//   else {
//     // printf("PC:DUT:%lx -> REF:%lx\n", ((CPU_state *)dut)->pc, cpu.pc);
//     for (int i = 0; i < 32; ++i) { cpu.gpr[i] = ((CPU_state *)dut)->gpr[i]; }
//     cpu.pc = ((CPU_state *)dut)->pc;
//     for (int i = 0; i <  4; ++i) { cpu.csr[i] = ((CPU_state *)dut)->csr[i]; }

//   }
// }

void difftest_exec(uint64_t n) {
  // Get the toy extension instance
  toy_t *toy = toy_t::get_instance();
  if (!toy) {
    printf("Error: toy extension not initialized\n");
    return;
  }

  // Get the processor from toy extension
  processor_t *processor = toy->get_processor();
  if (!processor) {
    printf("Error: processor not available in toy extension\n");
    return;
  }

  // Execute n steps on the spike processor
  processor->step(n);
}
