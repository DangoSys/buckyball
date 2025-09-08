#ifndef _BUCKYBALL_H
#define _BUCKYBALL_H

#include <riscv/extension.h>
#include <riscv/rocc.h>
#include <vector>
#include "buckyball_params.h"

static const uint32_t sp_matrices = (BANK_NUM * BANK_ROWS) / DIM;
static const uint64_t spAddrLen = SPAD_ADDR_LEN;
static const uint64_t memAddrLen = MEM_ADDR_LEN;

#define MAKECUSTOMFN(opcode) custom ## opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

struct buckyball_state_t {
  void reset();

  bool enable;
  bool resetted = false;

  std::vector<std::vector<elem_t>> spad; // Scratchpad only
};

class toy_t : public rocc_t {
public:
  toy_t() {}
  const char* name() const { return "toy"; }

  reg_t custom3(processor_t *p, rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void set_processor(processor_t* p) { this->p = p; }

  void mvin(reg_t dram_addr, reg_t sp_addr);
  void mvout(reg_t dram_addr, reg_t sp_addr);
  void mul_warp16(reg_t rs1, reg_t rs2);
  void bbfp_mul(reg_t rs1, reg_t rs2);

private:
  buckyball_state_t buckyball_state;
  processor_t* p;

  const unsigned mvin_funct = 24;   // func7: 0010000
  const unsigned mvout_funct = 25;  // func7: 0010001
  const unsigned mul_funct = 32; // func7: 0100000 (bb_mul_warp16)
  const unsigned fence_funct = 31;
  const unsigned bbfp_mul_funct = 26;   // func7: 0100001
  const unsigned matmul_ws_funct = 27;   // func7: 0100010
  template <class T>
  T read_from_dram(reg_t addr);

  template <class T>
  void write_to_dram(reg_t addr, T data);
};

#endif
