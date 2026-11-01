// One +CASE program per assertion test. The thread runs in machine mode and implements
// only `csrr rd, mcycle`; every other CSR access traps as illegal, as do EBREAK and an
// ECALL with a non-zero service number. The run ends at the RTL assertion.
task automatic illegal(string name);
  logic [63:0] code;
  case (name)
    "csr_mstatus": code = {32'h00000073, 32'h30002573};  // csrr a0, mstatus
    "csr_cycle": code = {32'h00000073, 32'hc0002573};  // csrr a0, cycle
    "csr_minstret": code = {32'h00000073, 32'hb0202573};  // csrr a0, minstret
    "csr_write_mcycle": code = {32'h00000073, 32'hb0051073};  // csrw mcycle, a0
    "csr_set_mcycle": code = {32'h00000073, 32'hb005a573};  // csrrs a0, mcycle, a1
    "csr_clear_mcycle": code = {32'h00000073, 32'hb0003573};  // csrrc a0, mcycle, x0
    "csr_immediate_mcycle": code = {32'h00000073, 32'hb0006573};  // csrrsi a0, mcycle, 0
    "ebreak": code = {32'h00000073, 32'h00100073};
    "ecall_service": code = {32'h00000073, 32'h00100893};  // li a7, 1; ecall
    default: `uvm_fatal("CASE", $sformatf("unknown case %s", name))
  endcase
  put(0, 0, {64'b0, code}, 1);
  code_end[0] = 8;
  in_use = 1;
  launch(0);
  repeat (100) @(negedge clock);
endtask
