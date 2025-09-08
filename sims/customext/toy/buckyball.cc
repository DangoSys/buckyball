#include "buckyball.h"
#include "buckyball_params.h"
#include <cstdio>
#include <riscv/mmu.h>
#include <riscv/trap.h>
#include <cassert>

using namespace std;

REGISTER_EXTENSION(toy, []() { return new toy_t; })

#define dprintf(...) { if (p->get_log_commits_enabled()) printf(__VA_ARGS__); }

void buckyball_state_t::reset() {
  enable = true;
  
  spad.clear();
  spad.resize(sp_matrices*DIM, std::vector<elem_t>(DIM, 0));
  
  resetted = true;
  
  // printf("buckyball extension configured with:\n");
  // printf("    dim = %u\n", DIM);
}


template <class T>
T toy_t::read_from_dram(reg_t addr) {
  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    value |= p->get_mmu()->load<uint8_t>(addr + byte_idx) << (byte_idx*8);
  }
  return value;
}

template <class T>
void toy_t::write_to_dram(reg_t addr, T data) {
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    p->get_mmu()->store<uint8_t>(addr + byte_idx, (data >> (byte_idx*8)) & 0xFF);
  }
}

// Move data from DRAM to scratchpad
// rs1: mem_addr, rs2: sp_addr[spAddrLen-1:0] | rows[spAddrLen+9:spAddrLen]
// 每次都搬运完整的行(DIM个元素)
void toy_t::mvin(reg_t rs1, reg_t rs2) {
  auto const base_dram_addr = rs1 & ((1UL << memAddrLen) - 1); // rs1 memddrLen-1:0
  auto const base_sp_addr = rs2 & ((1UL << spAddrLen) - 1);  // rs2[spAddrLen-1:0]
  auto const rows = (rs2 >> spAddrLen) & 0x3FF;  // rs2[spAddrLen+9:spAddrLen], 10 bits
  
  dprintf("BUCKYBALL: mvin - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf("BUCKYBALL: mvin - 0x%02lx rows from mem 0x%08lx to spad 0x%08lx\n", 
          rows, base_dram_addr, base_sp_addr);
  
  for (size_t i = 0; i < rows; ++i) {
    auto const dram_row_addr = base_dram_addr + i*DIM*sizeof(elem_t);
    const size_t spad_row = base_sp_addr + i;

    for (size_t j = 0; j < DIM; ++j) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(elem_t);
      elem_t value = read_from_dram<elem_t>(dram_byte_addr);
      buckyball_state.spad.at(spad_row).at(j) = value;
      // dprintf("%d ", value);
    }
    // dprintf("\n");
  }
}

// Move data from scratchpad to DRAM  
// rs1: mem_addr, rs2: sp_addr[spAddrLen-1:0] | rows[spAddrLen+9:spAddrLen]
// 每次都搬运完整的行(DIM个元素)
void toy_t::mvout(reg_t rs1, reg_t rs2) {
  auto const base_dram_addr = rs1 & ((1UL << memAddrLen) - 1); // rs1 memddrLen-1:0
  auto const base_sp_addr = rs2 & ((1UL << spAddrLen) - 1);  // rs2[spAddrLen-1:0]
  auto const rows = (rs2 >> spAddrLen) & 0x3FF;  // rs2[spAddrLen+9:spAddrLen], 10 bits

  dprintf("BUCKYBALL: mvout - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf("BUCKYBALL: mvout - 0x%02lx rows from spad 0x%08lx to mem 0x%08lx\n", 
          rows, base_sp_addr, base_dram_addr);

  for (size_t i = 0; i < rows; ++i) {
    auto const dram_row_addr = base_dram_addr + i*DIM*sizeof(elem_t);
    const size_t spad_row = base_sp_addr + i;

    for (size_t j = 0; j < DIM; ++j) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(elem_t);
      elem_t value = buckyball_state.spad.at(spad_row).at(j);
      write_to_dram<elem_t>(dram_byte_addr, value);
      // dprintf("%d ", value);
    }
    // dprintf("\n");
  }
}

// Matrix multiplication using warp16 pattern
void toy_t::mul_warp16(reg_t rs1, reg_t rs2) {
  auto const op1_spaddr = rs1 & ((1UL << spAddrLen) - 1);  // rs1[spAddrLen-1:0]
  auto const op2_spaddr = (rs1 >> spAddrLen) & ((1UL << spAddrLen) - 1);  // rs1[2*spAddrLen-1:spAddrLen]
  auto const wr_spaddr = rs2 & ((1UL << spAddrLen) - 1);   // rs2[spAddrLen-1:0]  
  auto const iter = (rs2 >> spAddrLen) & 0x3FF;  // rs2[spAddrLen+9:spAddrLen], 10 bits

  // TODO:加个assert，op1_spaddr和op2_spaddr不能属于同一个bank

  dprintf("BUCKYBALL: mul_warp16 - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf("BUCKYBALL: mul_warp16 - op1_spaddr=0x%08lx, op2_spaddr=0x%08lx, wr_spaddr=0x%08lx, iter=0x%02lx\n", 
          op1_spaddr, op2_spaddr, wr_spaddr, iter);

  // Perform matrix multiplication for specified iterations
  for (size_t i = 0; i < iter; ++i) {
    // For each iteration, compute one row of result matrix
    const size_t result_row = wr_spaddr + i;
    const size_t op1_row = op1_spaddr + i;
    
    // Initialize result row to zero
    for (size_t col = 0; col < DIM; ++col) {
      buckyball_state.spad.at(result_row).at(col) = 0;
    }
    
    // Compute dot product for each column of result
    for (size_t col = 0; col < DIM; ++col) {
      elem_t sum = 0;
      for (size_t k = 0; k < DIM; ++k) {
        // op1[i][k] * op2[k][col]
        elem_t a = buckyball_state.spad.at(op1_row).at(k);
        elem_t b = buckyball_state.spad.at(op2_spaddr + k).at(col);
        sum += a * b;
      }
      buckyball_state.spad.at(result_row).at(col) = sum;
    }
  }
}

// Matrix multiplication using warp16 pattern
void toy_t::bbfp_mul(reg_t rs1, reg_t rs2) {
  auto const op1_spaddr = rs1 & ((1UL << spAddrLen) - 1);  // rs1[spAddrLen-1:0]
  auto const op2_spaddr = (rs1 >> spAddrLen) & ((1UL << spAddrLen) - 1);  // rs1[2*spAddrLen-1:spAddrLen]
  auto const wr_spaddr = rs2 & ((1UL << spAddrLen) - 1);   // rs2[spAddrLen-1:0]  
  auto const iter = (rs2 >> spAddrLen) & 0x3FF;  // rs2[spAddrLen+9:spAddrLen], 10 bits
  // scratchpad 行数
  // size_t spad_rows = buckyball_state.spad.size();

  // // 检查起始行号和迭代次数是否合法
  // assert(op1_spaddr + iter <= spad_rows && "op1_spaddr越界");
  // assert(op2_spaddr + iter <= spad_rows && "op2_spaddr越界");
  // assert(wr_spaddr + iter <= spad_rows && "wr_spaddr越界");

  // TODO:加个assert，op1_spaddr和op2_spaddr不能属于同一个bank

  dprintf("BUCKYBALL: bbfp_mul - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf("BUCKYBALL: bbfp_mul - op1_spaddr=0x%08lx, op2_spaddr=0x%08lx, wr_spaddr=0x%08lx, iter=0x%02lx\n", 
          op1_spaddr, op2_spaddr, wr_spaddr, iter);
  dprintf("BBFP_MUl_Test\n");

  // Perform matrix multiplication for specified iterations
  for (size_t i = 0; i < iter; ++i) {
    // For each iteration, compute one row of result matrix
    const size_t result_row = wr_spaddr + i;
    const size_t op1_row = op1_spaddr + i;
    const size_t op2_row = op2_spaddr + i;
    elem_t share_exp_a = buckyball_state.spad.at(op1_row).at(0);
    elem_t share_exp_b = buckyball_state.spad.at(op2_row).at(0);
    // Initialize result row to zero
    for (size_t col = 0; col < DIM; ++col) {
      buckyball_state.spad.at(result_row).at(col) = 0;
    }
    
    elem_t sum_exp = share_exp_a + share_exp_b;
    buckyball_state.spad.at(result_row).at(0) = sum_exp;
    // Compute dot product for each column of result
    for (size_t col = 1; col < DIM; ++col) {
      elem_t sum = 0;
      for (size_t k = 1; k < DIM; ++k) {
        // op1[i][k] * op2[k][col]
        elem_t a = buckyball_state.spad.at(op1_row).at(k);
        elem_t b = buckyball_state.spad.at(op2_row).at(k);
        sum += a * b;
      }
      buckyball_state.spad.at(result_row).at(col) = sum;
    }
  }
}


reg_t toy_t::custom3(processor_t *p, rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  this->p = p; // Store the processor pointer for use in other methods
  
  if (!buckyball_state.resetted) {
    buckyball_state.reset();
  }

  if (insn.funct == mvin_funct) {
    mvin(xs1, xs2);
  } else if (insn.funct == mvout_funct) {
    mvout(xs1, xs2);
  } else if (insn.funct == mul_funct) {
    mul_warp16(xs1, xs2);
  } else if (insn.funct == fence_funct) {
    dprintf("BUCKYBALL: flush\n");
  } else if (insn.funct == bbfp_mul_funct) {
    bbfp_mul(xs1, xs2);
  } 
  else {
    dprintf("BUCKYBALL: encountered unknown instruction with funct: %d\n", insn.funct);
    illegal_instruction(*p);
  }
  
  return 0;
}
