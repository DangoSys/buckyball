#ifndef _TOY_H
#define _TOY_H

#include "common.h"
#include "params.h"
#include <riscv/extension.h>
#include <riscv/rocc.h>
#include <vector>

static const uint32_t sp_matrices = (BANK_NUM * BANK_ROWS) / DIM;
static const uint32_t acc_matrices = (BANK_NUM * BANK_ROWS) / DIM;
static const uint64_t spAddrLen = SPAD_ADDR_LEN;
static const uint64_t memAddrLen = MEM_ADDR_LEN;

#define MAKECUSTOMFN(opcode) custom##opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

struct toy_state_t {
  void reset();

  bool enable;
  bool resetted = false;

  std::vector<std::vector<elem_t>> spad; // Scratchpad
  std::vector<std::vector<acc_t>> acc;   // Accumulator
  std::vector<std::vector<int32_t>> rf;  // Register files for indices
};

class toy_t : public rocc_t {
public:
  toy_t() { instance = this; }
  const char *name() const { return "toy"; }

  virtual reg_t custom3(processor_t *, rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void set_processor(processor_t *p) { this->p = p; }

  // Static method to get the current instance
  static toy_t *get_instance() { return instance; }
  processor_t *get_processor() { return p; }

  std::vector<insn_desc_t> get_instructions(const processor_t &proc);
  std::vector<disasm_insn_t *> get_disasms(const processor_t *proc);

  void mvin(reg_t dram_addr, reg_t sp_addr);
  void mvout(reg_t dram_addr, reg_t sp_addr);
  void mul_warp16(reg_t rs1, reg_t rs2);
  void bbfp_mul(reg_t rs1, reg_t rs2);
  void sparse_mul(reg_t rs1, reg_t rs2);
  void scatter_mvin(reg_t rs1, reg_t rs2);

private:
  toy_state_t toy_state;
  processor_t *p;

  // Static instance for global access
  static toy_t *instance;

  const unsigned mvin_funct = 24;  // func7: 0010000
  const unsigned mvout_funct = 25; // func7: 0010001
  const unsigned mul_funct = 32;   // func7: 0100000 (bb_mul_warp16)
  const unsigned fence_funct = 31;
  const unsigned bbfp_mul_funct = 26;     // func7: 0100001
  const unsigned matmul_ws_funct = 27;    // func7: 0100010
  const unsigned sparse_mul_funct = 33;   // func7: 0100001 (bb_sparse_mul)
  const unsigned scatter_mvin_funct = 34; // func7: 0100010 (bb_scatter_mvin)
};

#endif // _TOY_H
