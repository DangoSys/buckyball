#include "toy.h"
#include "socket.h"
#include <cassert>
#include <cstdio>
#include <riscv/mmu.h>
#include <riscv/trap.h>

using namespace std;

REGISTER_EXTENSION(toy, []() { return new toy_t; })

toy_t::toy_t() : socket_client(new SocketClient()) {}

toy_t::~toy_t() {
  // socket_client will be automatically destroyed
}

#define dprintf(...)                                                           \
  {                                                                            \
    if (p->get_log_commits_enabled())                                          \
      printf(__VA_ARGS__);                                                     \
  }

template <class T> T toy_t::read_from_dram(reg_t addr) {
  T value = 0;
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    value |= p->get_mmu()->load<uint8_t>(addr + byte_idx) << (byte_idx * 8);
  }
  return value;
}

template <class T> void toy_t::write_to_dram(reg_t addr, T data) {
  for (size_t byte_idx = 0; byte_idx < sizeof(T); ++byte_idx) {
    p->get_mmu()->store<uint8_t>(addr + byte_idx,
                                 (data >> (byte_idx * 8)) & 0xFF);
  }
}

void toy_state_t::reset() {
  enable = true;
  resetted = true;
}

reg_t toy_t::CUSTOMFN(XCUSTOM_ACC)(rocc_insn_t insn, reg_t xs1, reg_t xs2) {

  if (!toy_state.resetted) {
    toy_state.reset();
  }

  // Set processor for socket client (for DMA operations)
  socket_client->set_processor(p);

  // Send socket request and wait for response
  dprintf("TOY: Processing custom instruction with funct=%d\n", insn.funct);
  reg_t result = socket_client->send_and_wait(insn.funct, xs1, xs2);

  dprintf("TOY: custom instruction funct=%d completed with result=0x%lx\n",
          insn.funct, result);

  return result;
}

static reg_t toy_custom(processor_t *p, insn_t insn, reg_t pc) {
  toy_t *toy = static_cast<toy_t *>(p->get_extension("toy"));
  rocc_insn_union_t u;
  state_t *state = p->get_state();
  toy->set_processor(p);
  u.i = insn;
  reg_t xs1 = u.r.xs1 ? state->XPR[insn.rs1()] : -1;
  reg_t xs2 = u.r.xs2 ? state->XPR[insn.rs2()] : -1;
  reg_t xd = toy->CUSTOMFN(XCUSTOM_ACC)(u.r, xs1, xs2);
  if (u.r.xd) {
    state->log_reg_write[insn.rd() << 4] = {xd, 0};
    state->XPR.write(insn.rd(), xd);
  }
  return pc + 4;
}

std::vector<insn_desc_t> toy_t::get_instructions() {
  std::vector<insn_desc_t> insns;
  push_custom_insn(insns, ROCC_OPCODE3, ROCC_OPCODE_MASK, ILLEGAL_INSN_FUNC,
                   toy_custom);
  return insns;
}

std::vector<disasm_insn_t *> toy_t::get_disasms() {
  std::vector<disasm_insn_t *> insns;
  return insns;
}
