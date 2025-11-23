#ifndef _BEBOP_H
#define _BEBOP_H

#include "common.h"
#include <memory>
#include <riscv/extension.h>
#include <riscv/rocc.h>
#include <vector>

#define MAKECUSTOMFN(opcode) custom##opcode
#define CUSTOMFN(opcode) MAKECUSTOMFN(opcode)

// Forward declaration
class SocketClient;

struct bebop_state_t {
  void reset();
  bool enable;
  bool resetted = false;
};

class bebop_t : public extension_t {
public:
  bebop_t();
  ~bebop_t();
  const char *name() const override { return "bebop"; }

  reg_t CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2);
  void set_processor(processor_t *p) { this->p = p; }
  std::vector<insn_desc_t> get_instructions(const processor_t &proc) override;
  std::vector<disasm_insn_t *>
  get_disasms(const processor_t *proc = nullptr) override;

private:
  bebop_state_t bebop_state;
  processor_t *p;

  // Socket client
  std::unique_ptr<SocketClient> socket_client;
  template <class T> T read_from_dram(reg_t addr);
  template <class T> void write_to_dram(reg_t addr, T data);
};

#endif // _BEBOP_H
