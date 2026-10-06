`include "core_config.svh"
`ifdef IRQ_PROFILE
`define CORE_TB irq_tb
`define CORE_DUT Irq
`elsif SUPERVISOR_PROFILE
`define CORE_TB supervisor_tb
`define CORE_DUT Supervisor
`elsif FPU_PROFILE
`define CORE_TB fpu_tb
`define CORE_DUT Fpu
`else
`define CORE_TB core_tb
`define CORE_DUT Core
`endif
module `CORE_TB;
  import uvm_pkg::*;
  import core_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  logic [63:0] time_value = 0;
  always @(posedge clock) time_value <= control.reset ? 0 : time_value + 1;
  core_control_if control (clock);
  stream_if #(`CORE_MEMREQ_WIDTH) mem_req (
      clock,
      control.reset
  );
  stream_if #(`CORE_MEMRESP_WIDTH) mem_resp (
      clock,
      control.reset
  );
  stream_if #(`CORE_UNCACHED_WIDTH) uncached_req (
      clock,
      control.reset
  );
  stream_if #(`CORE_URESP_WIDTH) uncached_resp (
      clock,
      control.reset
  );
  `CORE_DUT dut (
      .clock(clock),
      .reset(control.reset),
      .io_resetVector(64'h80000000),
      .io_time(time_value),
`ifdef IRQ_PROFILE
      .io_timerInterrupt(control.timer_irq),
      .io_softwareInterrupt(control.software_irq),
      .io_externalInterrupt(control.external_irq),
`else
      .io_timerInterrupt(1'b0),
      .io_softwareInterrupt(1'b0),
      .io_externalInterrupt(1'b0),
`endif
      .io_retired(control.retired),
      .io_retiredPc(control.retired_pc),
      .io_trapped(control.trapped),
      .io_trapCause(control.trap_cause),
      .io_trapValue(control.trap_value),
      .io_trapPc(control.trap_pc),
      .io_cancelledData(control.cancelled),
      .io_memoryOutstanding(control.outstanding),
      `include "core_ports.svh"
  );
`ifdef IRQ_PROFILE
  `include "irq_fixture.svh"
  wire demand_fire = dut.core.cache.io_access_valid && dut.core.cache.io_access_ready;
  assign control.target_access[0] = demand_fire && dut.core.cache.io_access_bits_addr == 44'h800040c0 && !dut.core.cache.io_access_bits_write && dut.core.cache.io_access_bits_atomic == 0;
  assign control.target_access[1] = demand_fire && dut.core.cache.io_access_bits_addr == 44'h800040c8 && dut.core.cache.io_access_bits_atomic == 2;
  assign control.target_access[2] = demand_fire && dut.core.cache.io_access_bits_addr == 44'h800040d0 && dut.core.cache.io_access_bits_write && dut.core.cache.io_access_bits_atomic == 0;
  assign control.load_waiting = dut.core.lsu.io_response_ready && dut.core.lsu.instructionPc == `IRQ_LOAD_PC;
  assign control.amo_completed = dut.core.lsu.io_cpu_req_ready && !dut.core.lsu.io_idle && dut.core.lsu.instructionPc == `IRQ_AMO_PC;
  assign control.store_completed = dut.core.lsu.io_cpu_req_ready && !dut.core.lsu.io_idle && dut.core.lsu.instructionPc == `IRQ_STORE_PC;
`endif
  initial begin
    uvm_config_db#(virtual core_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`CORE_MEMREQ_WIDTH))::set(null, "uvm_test_top*", "mem_req",
                                                                 mem_req);
    uvm_config_db#(virtual stream_if #(`CORE_MEMRESP_WIDTH))::set(null, "uvm_test_top*", "mem_resp",
                                                                  mem_resp);
    uvm_config_db#(virtual stream_if #(`CORE_UNCACHED_WIDTH))::set(null, "uvm_test_top*",
                                                                   "uncached_req", uncached_req);
    uvm_config_db#(virtual stream_if #(`CORE_URESP_WIDTH))::set(null, "uvm_test_top*",
                                                                "uncached_resp", uncached_resp);
    run_test("protocol_test");
  end
endmodule
