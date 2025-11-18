#ifndef _TOY_H
#define _TOY_H

#include "common.h"
#include <memory>
#include <riscv/extension.h>
#include <riscv/rocc.h>
#include <vector>

#define MAKECUSTOMFN(opcode) custom##opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

// Forward declaration
class SocketClient;

struct toy_state_t {
  void reset();
  bool enable;
  bool resetted = false;
};

class toy_t : public extension_t {
public:
  toy_t();
  ~toy_t();
  const char *name() { return "toy"; }

  reg_t CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void set_processor(processor_t *p) { this->p = p; }
  // void rocc(reg_t rs1, reg_t rs2);
  // void dma_read(reg_t dram_addr);
  // void dma_write(reg_t dram_addr);
  std::vector<insn_desc_t> get_instructions();
  std::vector<disasm_insn_t *> get_disasms();

private:
  toy_state_t toy_state;
  processor_t *p;

  // Socket client
  std::unique_ptr<SocketClient> socket_client;

  template <class T> T read_from_dram(reg_t addr);

  template <class T> void write_to_dram(reg_t addr, T data);
};

#endif // _TOY_H
