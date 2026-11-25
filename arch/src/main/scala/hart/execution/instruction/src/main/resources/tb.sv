module instruction_tb;
  import uvm_pkg::*;
  import instruction_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  clock_if timing (clock);
  port_if bus (
      clock,
      timing.reset
  );
  Instruction dut (
      .clock(clock),
      .reset(bus.reset),
      .io_bits(bus.bits),
      .io_expanded(bus.expanded),
      .io_compressed(bus.compressed),
      .io_illegal(bus.illegal),
      .io_opcode(bus.opcode),
      .io_funct3(bus.funct3),
      .io_funct7(bus.funct7),
      .io_rs1(bus.rs1),
      .io_rs2(bus.rs2),
      .io_rd(bus.rd),
      .io_csrAddress(bus.csr_address),
      .io_immediate(bus.immediate),
      .io_atomicCode(bus.atomic),
      .io_integerFunction(bus.function_code),
      .io_integerWord(bus.word),
      .io_integerImmediate(bus.immediate_mode)
  );
  initial begin
    uvm_config_db#(virtual clock_if)::set(null, "uvm_test_top*", "timing", timing);
    uvm_config_db#(virtual port_if)::set(null, "uvm_test_top*", "vif", bus);
    run_test("protocol_test");
  end
endmodule
