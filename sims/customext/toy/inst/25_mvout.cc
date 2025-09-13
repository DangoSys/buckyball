#include "common.h"
#include "inst.h"
#include "toy.h"
#include <param/dma.h>

// Move data from scratchpad to DRAM
// rs1: mem_addr, rs2: sp_addr[spAddrLen-1:0] | rows[spAddrLen+9:spAddrLen]
// 按bank行宽度搬运数据
void toy_t::mvout(reg_t rs1, reg_t rs2) {
  mvin_rs1_t rs1_fields(rs1);
  mvin_rs2_t rs2_fields(rs2);

  auto const base_dram_addr = rs1_fields.base_dram_addr();
  auto const base_sp_addr = rs2_fields.base_sp_addr();
  auto const iter = rs2_fields.iter();

  uint32_t bank = spad_get_bank(base_sp_addr);
  uint32_t start_offset = spad_get_offset(base_sp_addr);
  uint32_t row_bytes = spad_get_bank_row_bytes(bank);

  dprintf(p, "TOY: mvout - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf(p,
          "TOY: mvout - %02d times dma write from spad 0x%08lx (bank=%d, "
          "row=%d) to mem 0x%08lx\n",
          iter, base_sp_addr, bank, start_offset, base_dram_addr);

  uint32_t dma_bytes = dma_row_bytes();

  uint32_t elem_size = spad_get_bank_row_bytes(bank) / DIM;

  for (size_t i = 0; i < iter; ++i) {
    for (size_t j = 0; j < DIM; ++j) {
      auto const dram_addr =
          base_dram_addr + i * DIM * elem_size + j * elem_size;

      if (elem_size == 1) {
        elem_t value =
            read_from_bank<elem_t>(toy_state, bank, start_offset + i, j);
        write_to_dram<elem_t>(p, dram_addr, value);
      } else {
        acc_t value =
            read_from_bank<acc_t>(toy_state, bank, start_offset + i, j);
        write_to_dram<acc_t>(p, dram_addr, value);
      }
    }
  }
}
