`include "lsu_config.svh"
module lsu_tb;
  import uvm_pkg::*;
  import lsu_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  lsu_control_if control (clock);
  // EX structural replay may withdraw an unaccepted HellaCache offer; backend streams remain irrevocable.
  stream_if #(`CORE_CPU_WIDTH) source_if (
      clock,
      control.reset || control.cancel_offer
  );
  stream_if #(`CORE_RETURN_WIDTH) sink_if (
      clock,
      control.reset
  );
  stream_if #(`CORE_VIRTUAL_WIDTH) virtual_req (
      clock,
      control.reset
  );
  stream_if #(`CORE_RESULT_WIDTH) virtual_resp (
      clock,
      control.reset
  );
  assign sink_if.ready = 1;
  Lsu dut (
      .clock(clock),
      .reset(control.reset),
      .io_pc(control.pc),
      .io_cpu_s1_kill(control.s1_kill),
      .io_cpu_s2_kill(control.s2_kill),
      .io_cpu_s1_data_data(control.s1_data),
      .io_cpu_s1_data_mask(8'hff),
      .io_cpu_keep_clock_enabled(1'b1),
      .io_cpu_s2_nack(control.nack),
      .io_cpu_ordered(control.ordered),
      .io_idle(control.idle),
      .io_cancelled(control.cancelled),
      .io_cpu_s2_xcpt_ma_ld(control.ma_ld),
      .io_cpu_s2_xcpt_ma_st(control.ma_st),
      .io_cpu_s2_xcpt_pf_ld(control.pf_ld),
      .io_cpu_s2_xcpt_pf_st(control.pf_st),
      .io_cpu_s2_xcpt_ae_ld(control.ae_ld),
      .io_cpu_s2_xcpt_ae_st(control.ae_st),
      `include "lsu_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual lsu_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`CORE_CPU_WIDTH))::set(null, "uvm_test_top*", "cpu",
                                                              source_if);
    uvm_config_db#(virtual stream_if #(`CORE_RETURN_WIDTH))::set(null, "uvm_test_top*", "result",
                                                                 sink_if);
    uvm_config_db#(virtual stream_if #(`CORE_VIRTUAL_WIDTH))::set(null, "uvm_test_top*",
                                                                  "virtual_req", virtual_req);
    uvm_config_db#(virtual stream_if #(`CORE_RESULT_WIDTH))::set(null, "uvm_test_top*",
                                                                 "virtual_resp", virtual_resp);
    run_test("protocol_test");
  end
endmodule
