#include "common.h"
#include "inst.h"
#include "toy.h"

// Move data from DRAM to scratchpad
// rs1: mem_addr, rs2: sp_addr[spAddrLen-1:0] | rows[spAddrLen+9:spAddrLen]
// 每次都搬运完整的行(DIM个元素)
void toy_t::mvin(reg_t rs1, reg_t rs2) {
  // rs1 memddrLen-1:0
  mvin_rs1_t rs1_fields(rs1);
  mvin_rs2_t rs2_fields(rs2);

  auto const base_dram_addr = rs1_fields.base_dram_addr();
  auto const base_sp_addr = rs2_fields.base_sp_addr();
  auto const rows = rs2_fields.rows();

  dprintf(p, "TOY: mvin - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf(p, "TOY: mvin - 0x%02lx rows from mem 0x%08lx to spad 0x%08lx\n",
          rows, base_dram_addr, base_sp_addr);

  for (size_t i = 0; i < rows; ++i) {
    auto const dram_row_addr = base_dram_addr + i * DIM * sizeof(elem_t);
    const size_t spad_row = base_sp_addr + i;

    for (size_t j = 0; j < DIM; ++j) {
      auto const dram_byte_addr = dram_row_addr + j * sizeof(elem_t);
      elem_t value = read_from_dram<elem_t>(p, dram_byte_addr);
      toy_state.spad.at(spad_row).at(j) = value;
      // dprintf(p, "%d ", value);
    }
    // dprintf(p, "\n");
  }
}
