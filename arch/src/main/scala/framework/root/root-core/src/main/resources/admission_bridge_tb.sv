`include "admission_bridge_config.svh"
`ifndef AB_TB
`define AB_TB admission_bridge_tb
`endif
module `AB_TB;
  import uvm_pkg::*;
  import admission_bridge_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  admission_bridge_control_if control (clock);
  stream_if #(`AB_SNAP_WIDTH) command (
      clock,
      control.reset
  );
  stream_if #(`AB_RET_WIDTH) retirement (
      clock,
      control.reset
  );
  stream_if #(`AB_NPU_WIDTH) npu (
      clock,
      control.reset
  );
  assign control.allocation_valid=control.boot_allocation ||
   (npu.valid&&npu.ready&&npu.bits[`AB_NPU_FUNCT_OFFSET+:`AB_NPU_FUNCT_WIDTH]>1);
  AdmissionBridge dut (
      .clock(clock),
      .reset(control.reset),
      `include "admission_bridge_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual admission_bridge_control_if)::set(null, "uvm_test_top", "control",
                                                             control);
    uvm_config_db#(virtual stream_if #(`AB_SNAP_WIDTH))::set(null, "uvm_test_top", "command",
                                                             command);
    uvm_config_db#(virtual stream_if #(`AB_RET_WIDTH))::set(null, "uvm_test_top", "retirement",
                                                            retirement);
    uvm_config_db#(virtual stream_if #(`AB_NPU_WIDTH))::set(null, "uvm_test_top", "npu", npu);
`ifdef AB_LATE_FAULT
    uvm_config_db#(bit)::set(null, "uvm_test_top", "late_fault", 1);
`endif
    run_test("protocol_test");
  end
endmodule
