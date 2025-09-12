#include "toy.h"
#include "params.h"
#include <cassert>
#include <cstdio>
#include <riscv/extension.h>
#include <riscv/mmu.h>
#include <riscv/trap.h>

// Instruction implementations are now in separate source files

using namespace std;

#define dprintf(p, ...)                                                        \
  {                                                                            \
    if (p->get_log_commits_enabled())                                          \
      printf(__VA_ARGS__);                                                     \
  }

// Define static instance variable
toy_t *toy_t::instance = nullptr;

REGISTER_EXTENSION(toy, []() { return new toy_t; })

void toy_state_t::reset() {
  enable = true;

  spad.clear();
  acc.clear();
  rf.clear();
  spad.resize(sp_matrices * DIM, std::vector<elem_t>(DIM, 0));
  acc.resize(acc_matrices * DIM, std::vector<acc_t>(DIM, 0));
  rf.resize(2,
            std::vector<int32_t>(DIM * DIM, 0)); // 2 banks for register files

  resetted = true;

  // printf("toy extension configured with:\n");
  // printf("    dim = %u\n", DIM);
}

reg_t toy_t::custom3(processor_t *p, rocc_insn_t insn, reg_t xs1, reg_t xs2) {
  // Store the processor pointer for this call
  this->p = p;

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
    dprintf(p, "TOY: flush\n");
  } else if (insn.funct == bbfp_mul_funct) {
    bbfp_mul(xs1, xs2);
  } else if (insn.funct == scatter_mvin_funct) {
    scatter_mvin(xs1, xs2);
  } else if (insn.funct == sparse_mul_funct) {
    sparse_mul(xs1, xs2);
  } else {
    dprintf(p, "TOY: encountered unknown instruction with funct: %d\n",
            insn.funct);
    // Use the global illegal_instruction function for now
    throw trap_illegal_instruction(0);
  }

  return 0;
}

// 覆写custom3函数，因为custom3函数需要processor_t指针，而rocc_t没有这个指针
static reg_t toy_custom3(processor_t *p, insn_t insn, reg_t pc) {
  toy_t *rocc = static_cast<toy_t *>(p->get_extension("toy"));
  rocc_insn_union_t u;
  state_t *state = p->get_state();
  u.i = insn;
  reg_t xs1 = u.r.xs1 ? state->XPR[insn.rs1()] : -1;
  reg_t xs2 = u.r.xs2 ? state->XPR[insn.rs2()] : -1;

  // Call custom3 with processor pointer
  reg_t xd = rocc->custom3(p, u.r, xs1, xs2);

  if (u.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {xd, 0};
    state->XPR.write(insn.rd(), xd);
  }
  return pc + 4;
}

std::vector<insn_desc_t> toy_t::get_instructions(const processor_t &proc) {
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE3, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC,
                   toy_custom3);
  return insns;
}

std::vector<disasm_insn_t *> toy_t::get_disasms(const processor_t *proc) {
  std::vector<disasm_insn_t *> insns;

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
  insns.push_back(new disasm_insn_t("toy_mvin", ROCC_OPCODE3 | (24 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25),
                                    {&toy_rs1, &toy_rs2}));

  // MVOUT instruction (funct = 25)
  insns.push_back(new disasm_insn_t("toy_mvout", ROCC_OPCODE3 | (25 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25),
                                    {&toy_rs1, &toy_rs2}));

  // MATMUL instruction (funct = 32)
  insns.push_back(new disasm_insn_t("toy_mul_warp16", ROCC_OPCODE3 | (32 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25),
                                    {&toy_rs1, &toy_rs2}));

  // BBFP_MATMUL instruction (funct = 26)

  insns.push_back(new disasm_insn_t("toy_bbfp_mul", ROCC_OPCODE3 | (26 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25),
                                    {&toy_rs1, &toy_rs2}));

  // MATMUL_WS instruction (funct = 27)
  insns.push_back(new disasm_insn_t("toy_matmul_ws", ROCC_OPCODE3 | (27 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25),
                                    {&toy_rs1, &toy_rs2}));

  // SCATTER_MVIN instruction (funct = 34)
  insns.push_back(
      new disasm_insn_t("toy_scatter_mvin", ROCC_OPCODE3 | (34 << 25),
                        ROCC_OPCODE_MASK | (0x7F << 25), {&toy_rs1, &toy_rs2}));

  // SPARSE_MUL instruction (funct = 33)
  insns.push_back(new disasm_insn_t("toy_sparse_mul", ROCC_OPCODE3 | (33 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25),
                                    {&toy_rs1, &toy_rs2}));

  // FENCE instruction (funct = 31) - no operands needed
  insns.push_back(new disasm_insn_t("toy_fence", ROCC_OPCODE3 | (31 << 25),
                                    ROCC_OPCODE_MASK | (0x7F << 25), {}));

  return insns;
}
