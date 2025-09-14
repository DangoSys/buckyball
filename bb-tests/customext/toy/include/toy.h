#ifndef _TOY_H
#define _TOY_H

#include "common.h"
#include "params.h"
#include <riscv/extension.h>
#include <riscv/rocc.h>
#include <vector>

// Include bbhw library
#include <bbhw/isa/isa.h>
#include <bbhw/mem/spad.h>

// static const uint32_t sp_matrices = (SPAD_BANK_NUM * SPAD_BANK_ROWS) / DIM;
// static const uint32_t acc_matrices = (ACC_BANK_NUM * ACC_BANK_ROWS) / DIM;
// static const uint64_t spAddrLen = SPAD_ADDR_LEN;
// static const uint64_t memAddrLen = MEM_ADDR_LEN;

#define MAKECUSTOMFN(opcode) custom##opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

struct toy_state_t {
  void reset();

  bool enable;
  bool resetted = false;

  // 按bank组织的统一spad存储
  // banks[bank_id][row][col] - 每个bank可以存储不同类型的数据
  std::vector<std::vector<std::vector<uint8_t>>>
      banks;                            // 统一的bank存储，用字节存储
  std::vector<std::vector<int32_t>> rf; // Register files for indices
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

  // Difftest接口 - 提供对内存的访问
  toy_state_t &get_toy_state() { return toy_state; }
  const toy_state_t &get_toy_state() const { return toy_state; }

  std::vector<insn_desc_t> get_instructions(const processor_t &proc);
  std::vector<disasm_insn_t *> get_disasms(const processor_t *proc);

  void mvin(reg_t dram_addr, reg_t sp_addr);
  void mvout(reg_t dram_addr, reg_t sp_addr);
  void mul_warp16(reg_t rs1, reg_t rs2);
  void bbfp_mul(reg_t rs1, reg_t rs2);
  void transpose(reg_t rs1, reg_t rs2);
  void im2col(reg_t rs1, reg_t rs2);

private:
  toy_state_t toy_state;
  processor_t *p;

  // Static instance for global access
  static toy_t *instance;

  // const unsigned mvin_funct = 24;  // func7: 0010000
  // const unsigned mvout_funct = 25; // func7: 0010001
  // const unsigned mul_funct = 32;   // func7: 0100000 (bb_mul_warp16)
  // const unsigned fence_funct = 31;
  // const unsigned bbfp_mul_funct = 26;  // func7: 0100001
  // const unsigned matmul_ws_funct = 27; // func7: 0100010
  // const unsigned transpose_funct = 33; // func7: 0100001 (bb_transpose)
  // const unsigned im2col_funct = 34;    // func7: 0100010 (bb_im2col)
};

// Universal bank access functions - use bank configuration to determine element
// size
template <class T>
T read_from_bank(toy_state_t &toy_state, uint32_t bank, uint32_t row,
                 uint32_t col) {
  uint32_t elem_size = spad_get_bank_row_bytes(bank) /
                       DIM; // Element size = row_bytes / elements_per_row
  uint32_t byte_offset = col * elem_size;

  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T) && byte_idx < elem_size;
       ++byte_idx) {
    if (byte_offset + byte_idx < toy_state.banks.at(bank).at(row).size()) {
      value |= static_cast<T>(
                   toy_state.banks.at(bank).at(row).at(byte_offset + byte_idx))
               << (byte_idx * 8);
    }
  }
  return value;
}

template <class T>
void write_to_bank(toy_state_t &toy_state, uint32_t bank, uint32_t row,
                   uint32_t col, T data) {
  uint32_t elem_size = spad_get_bank_row_bytes(bank) /
                       DIM; // Element size = row_bytes / elements_per_row
  uint32_t byte_offset = col * elem_size;

  for (size_t byte_idx = 0; byte_idx < sizeof(T) && byte_idx < elem_size;
       ++byte_idx) {
    if (byte_offset + byte_idx < toy_state.banks.at(bank).at(row).size()) {
      toy_state.banks.at(bank).at(row).at(byte_offset + byte_idx) =
          (data >> (byte_idx * 8)) & 0xFF;
    }
  }
}

#endif // _TOY_H
