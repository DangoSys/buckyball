#include "toy.h"
#include "toy_params.h"
#include <cstdio>
#include <riscv/mmu.h>
#include <riscv/trap.h>
#include <cassert>

using namespace std;

REGISTER_EXTENSION(toy, []() { return new toy_t; })

#define dprintf(...) { if (p->get_log_commits_enabled()) printf(__VA_ARGS__); }

void toy_state_t::reset() {
  enable = true;
  
  spad.clear();
  spad.resize(sp_matrices*DIM, std::vector<elem_t>(DIM, 0));
  
  resetted = true;
  
  // printf("toy extension configured with:\n");
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
  
  dprintf("TOY: mvin - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf("TOY: mvin - 0x%02lx rows from mem 0x%08lx to spad 0x%08lx\n", 
          rows, base_dram_addr, base_sp_addr);
  
  for (size_t i = 0; i < rows; ++i) {
    auto const dram_row_addr = base_dram_addr + i*DIM*sizeof(elem_t);
    const size_t spad_row = base_sp_addr + i;

    for (size_t j = 0; j < DIM; ++j) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(elem_t);
      elem_t value = read_from_dram<elem_t>(dram_byte_addr);
      toy_state.spad.at(spad_row).at(j) = value;
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

  dprintf("TOY: mvout - rs1=%lx, rs2=%lx\n", rs1, rs2);
  dprintf("TOY: mvout - 0x%02lx rows from spad 0x%08lx to mem 0x%08lx\n", 
          rows, base_sp_addr, base_dram_addr);

  for (size_t i = 0; i < rows; ++i) {
    auto const dram_row_addr = base_dram_addr + i*DIM*sizeof(elem_t);
    const size_t spad_row = base_sp_addr + i;

    for (size_t j = 0; j < DIM; ++j) {
      auto const dram_byte_addr = dram_row_addr + j*sizeof(elem_t);
      elem_t value = toy_state.spad.at(spad_row).at(j);
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

  dprintf("TOY: mul_warp16 - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf("TOY: mul_warp16 - op1_spaddr=0x%08lx, op2_spaddr=0x%08lx, wr_spaddr=0x%08lx, iter=0x%02lx\n", 
          op1_spaddr, op2_spaddr, wr_spaddr, iter);

  // Perform matrix multiplication for specified iterations
  for (size_t i = 0; i < iter; ++i) {
    // For each iteration, compute one row of result matrix
    const size_t result_row = wr_spaddr + i;
    const size_t op1_row = op1_spaddr + i;
    
    // Initialize result row to zero
    for (size_t col = 0; col < DIM; ++col) {
      toy_state.spad.at(result_row).at(col) = 0;
    }
    
    // Compute dot product for each column of result
    for (size_t col = 0; col < DIM; ++col) {
      elem_t sum = 0;
      for (size_t k = 0; k < DIM; ++k) {
        // op1[i][k] * op2[k][col]
        elem_t a = toy_state.spad.at(op1_row).at(k);
        elem_t b = toy_state.spad.at(op2_spaddr + k).at(col);
        sum += a * b;
      }
      toy_state.spad.at(result_row).at(col) = sum;
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

  dprintf("TOY: bbfp_mul - rs1=0x%08lx, rs2=0x%08lx\n", rs1, rs2);
  dprintf("TOY: bbfp_mul - op1_spaddr=0x%08lx, op2_spaddr=0x%08lx, wr_spaddr=0x%08lx, iter=0x%02lx\n", 
          op1_spaddr, op2_spaddr, wr_spaddr, iter);
  dprintf("BBFP_MUl_Test\n");

  // Perform matrix multiplication for specified iterations
  for (size_t i = 0; i < iter; ++i) {
    // For each iteration, compute one row of result matrix
    const size_t result_row = wr_spaddr + i;
    const size_t op1_row = op1_spaddr + i;
    const size_t op2_row = op2_spaddr + i;
    elem_t share_exp_a = toy_state.spad.at(op1_row).at(0);
    elem_t share_exp_b = toy_state.spad.at(op2_row).at(0);
    // Initialize result row to zero
    for (size_t col = 0; col < DIM; ++col) {
      toy_state.spad.at(result_row).at(col) = 0;
    }
    
    elem_t sum_exp = share_exp_a + share_exp_b;
    toy_state.spad.at(result_row).at(0) = sum_exp;
    // Compute dot product for each column of result
    for (size_t col = 1; col < DIM; ++col) {
      elem_t sum = 0;
      for (size_t k = 1; k < DIM; ++k) {
        // op1[i][k] * op2[k][col]
        elem_t a = toy_state.spad.at(op1_row).at(k);
        elem_t b = toy_state.spad.at(op2_row).at(k);
        sum += a * b;
      }
      toy_state.spad.at(result_row).at(col) = sum;
    }
  }
}


reg_t toy_t::custom3(rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  // Note: processor pointer is set by set_processor() method before calling this
  
  if (!toy_state.resetted) {
    toy_state.reset();
  }

  if (insn.funct == mvin_funct) {
    mvin(xs1, xs2);
  } else if (insn.funct == mvout_funct) {
    mvout(xs1, xs2);
  } else if (insn.funct == mul_funct) {
    mul_warp16(xs1, xs2);
  } else if (insn.funct == fence_funct) {
    dprintf("TOY: flush\n");
  } else if (insn.funct == bbfp_mul_funct) {
    bbfp_mul(xs1, xs2);
  } 
  else {
    dprintf("TOY: encountered unknown instruction with funct: %d\n", insn.funct);
    illegal_instruction(*this->p);
  }
  
  return 0;
}

// 覆写custom3函数，因为custom3函数需要processor_t指针，而rocc_t没有这个指针
static reg_t toy_custom3(processor_t* p, insn_t insn, reg_t pc) {
  toy_t* rocc = static_cast<toy_t*>(p->get_extension("toy"));
  rocc_insn_union_t u;
  state_t* state = p->get_state();
  u.i = insn;
  reg_t xs1 = u.r.xs1 ? state->XPR[insn.rs1()] : -1;
  reg_t xs2 = u.r.xs2 ? state->XPR[insn.rs2()] : -1;
  
  // Set processor pointer before calling custom3
  rocc->set_processor(p);
  reg_t xd = rocc->custom3(u.r, xs1, xs2);
  
  if (u.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {xd, 0};
    state->XPR.write(insn.rd(), xd);
  }
  return pc+4;
}

std::vector<insn_desc_t> toy_t::get_instructions(const processor_t &proc) {
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE3, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC, toy_custom3);
  return insns;
}

std::vector<disasm_insn_t*> toy_t::get_disasms(const processor_t *proc) {
  std::vector<disasm_insn_t*> insns;
  
  // Define argument types for toy instructions
  struct : public arg_t {
    std::string to_string(insn_t insn) const {
      return "x" + std::to_string(insn.rs1());
    }
  } static toy_rs1;
  
  struct : public arg_t {
    std::string to_string(insn_t insn) const {
      return "x" + std::to_string(insn.rs2());
    }
  } static toy_rs2;
  
  // Custom-3 opcode is ROCC_OPCODE3 (0111 1011)
  // MVIN instruction (funct = 24)
  insns.push_back(new disasm_insn_t("toy_mvin", 
    ROCC_OPCODE3 | (24 << 25), 
    ROCC_OPCODE_MASK | (0x7F << 25), 
    {&toy_rs1, &toy_rs2}));
  
  // MVOUT instruction (funct = 25)
  insns.push_back(new disasm_insn_t("toy_mvout", 
    ROCC_OPCODE3 | (25 << 25), 
    ROCC_OPCODE_MASK | (0x7F << 25), 
    {&toy_rs1, &toy_rs2}));
  
  // MATMUL instruction (funct = 32)
  insns.push_back(new disasm_insn_t("toy_mul_warp16", 
    ROCC_OPCODE3 | (32 << 25), 
    ROCC_OPCODE_MASK | (0x7F << 25), 
    {&toy_rs1, &toy_rs2}));

  // BBFP_MATMUL instruction (funct = 26)

  insns.push_back(new disasm_insn_t("toy_bbfp_mul", 
    ROCC_OPCODE3 | (26 << 25), 
    ROCC_OPCODE_MASK | (0x7F << 25), 
    {&toy_rs1, &toy_rs2}));

  // MATMUL_WS instruction (funct = 27)
  insns.push_back(new disasm_insn_t("toy_matmul_ws", 
    ROCC_OPCODE3 | (27 << 25), 
    ROCC_OPCODE_MASK | (0x7F << 25), 
    {&toy_rs1, &toy_rs2}));

  // FENCE instruction (funct = 31) - no operands needed
  insns.push_back(new disasm_insn_t("toy_fence", 
    ROCC_OPCODE3 | (31 << 25), 
    ROCC_OPCODE_MASK | (0x7F << 25), 
    {}));
  
  return insns;
}
