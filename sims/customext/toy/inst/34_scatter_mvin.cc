#include "common.h"
#include "inst.h"
#include "toy.h"

// Scatter move in for indices (load indices to register file)
// rs1: mem_addr, rs2: count[31:1] | rf_bank[0:0]
void toy_t::scatter_mvin(reg_t rs1, reg_t rs2) {
  scatter_mvin_rs1_t rs1_fields(rs1);
  scatter_mvin_rs2_t rs2_fields(rs2);

  auto const base_dram_addr = rs1_fields.base_dram_addr();
  auto const rf_bank = rs2_fields.rf_bank();
  auto const count = rs2_fields.count();

  dprintf(p, "TOY: scatter_mvin - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf(p, "TOY: scatter_mvin - %d indices from mem 0x%08x to RF bank %d\n",
          count, base_dram_addr, rf_bank);

  for (size_t i = 0; i < count; ++i) {
    auto const dram_byte_addr = base_dram_addr + i * sizeof(int32_t);
    int32_t value = read_from_dram<int32_t>(p, dram_byte_addr);
    toy_state.rf.at(rf_bank).at(i) = value;
    // dprintf(p, "RF[%d][%ld] = %d\n", rf_bank, i, value);
  }
}
