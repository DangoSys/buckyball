`include "ram_config.svh"
`ifdef RAM_DDR
`include "ram_ddr_config.svh"
`define RAM_TOP ram_ddr_tb
`define RAM_DUT RamDdr
`define RAM_PORTS "ram_ddr_ports.svh"
`else
`define RAM_TOP ram_tb
`define RAM_DUT Ram
`define RAM_PORTS "ram_ports.svh"
`endif
module `RAM_TOP;
  import uvm_pkg::*;
  import ram_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  ram_control_if control (clock);
  stream_if #(`RAM_CPU_WIDTH)
      cpu0 (
          clock,
          control.reset
      ),
      cpu1 (
          clock,
          control.reset
      );
  stream_if #(`RAM_RESULT_WIDTH)
      result0 (
          clock,
          control.reset
      ),
      result1 (
          clock,
          control.reset
      );
  stream_if #(`RAM_LINE_WIDTH)
      line0 (
          clock,
          control.reset
      ),
      line1 (
          clock,
          control.reset
      ),
      memory0 (
          clock,
          control.reset
      ),
      memory1 (
          clock,
          control.reset
      );
  stream_if #(`RAM_REPLY_WIDTH)
      reply0 (
          clock,
          control.reset
      ),
      reply1 (
          clock,
          control.reset
      ),
      returned0 (
          clock,
          control.reset
      ),
      returned1 (
          clock,
          control.reset
      );
`ifdef RAM_DDR
  stream_if #(`RAM_A_WIDTH)
      aw (
          clock,
          control.reset
      ),
      ar (
          clock,
          control.reset
      );
  stream_if #(`RAM_W_WIDTH) w (
      clock,
      control.reset
  );
  stream_if #(`RAM_B_WIDTH) b (
      clock,
      control.reset
  );
  stream_if #(`RAM_R_WIDTH) r (
      clock,
      control.reset
  );
`endif
  `RAM_DUT dut (
      .clock(clock),
      .reset(control.reset),
      `include `RAM_PORTS
  );
  initial begin
    uvm_config_db#(virtual ram_control_if)::set(null, "uvm_test_top", "control", control);
    uvm_config_db#(virtual stream_if #(`RAM_CPU_WIDTH))::set(null, "uvm_test_top", "cpu0", cpu0);
    uvm_config_db#(virtual stream_if #(`RAM_CPU_WIDTH))::set(null, "uvm_test_top", "cpu1", cpu1);
    uvm_config_db#(virtual stream_if #(`RAM_RESULT_WIDTH))::set(null, "uvm_test_top", "result0",
                                                                result0);
    uvm_config_db#(virtual stream_if #(`RAM_RESULT_WIDTH))::set(null, "uvm_test_top", "result1",
                                                                result1);
    uvm_config_db#(virtual stream_if #(`RAM_LINE_WIDTH))::set(null, "uvm_test_top", "line0", line0);
    uvm_config_db#(virtual stream_if #(`RAM_LINE_WIDTH))::set(null, "uvm_test_top", "line1", line1);
    uvm_config_db#(virtual stream_if #(`RAM_REPLY_WIDTH))::set(null, "uvm_test_top", "reply0",
                                                               reply0);
    uvm_config_db#(virtual stream_if #(`RAM_REPLY_WIDTH))::set(null, "uvm_test_top", "reply1",
                                                               reply1);
    uvm_config_db#(virtual stream_if #(`RAM_LINE_WIDTH))::set(null, "uvm_test_top", "memory0",
                                                              memory0);
    uvm_config_db#(virtual stream_if #(`RAM_LINE_WIDTH))::set(null, "uvm_test_top", "memory1",
                                                              memory1);
    uvm_config_db#(virtual stream_if #(`RAM_REPLY_WIDTH))::set(null, "uvm_test_top", "returned0",
                                                               returned0);
    uvm_config_db#(virtual stream_if #(`RAM_REPLY_WIDTH))::set(null, "uvm_test_top", "returned1",
                                                               returned1);
`ifdef RAM_DDR
    uvm_config_db#(virtual stream_if #(`RAM_A_WIDTH))::set(null, "uvm_test_top", "aw", aw);
    uvm_config_db#(virtual stream_if #(`RAM_A_WIDTH))::set(null, "uvm_test_top", "ar", ar);
    uvm_config_db#(virtual stream_if #(`RAM_W_WIDTH))::set(null, "uvm_test_top", "w", w);
    uvm_config_db#(virtual stream_if #(`RAM_B_WIDTH))::set(null, "uvm_test_top", "b", b);
    uvm_config_db#(virtual stream_if #(`RAM_R_WIDTH))::set(null, "uvm_test_top", "r", r);
`endif
    run_test("protocol_test");
  end
endmodule
