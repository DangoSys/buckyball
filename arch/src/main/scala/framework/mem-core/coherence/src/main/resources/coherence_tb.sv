`include "coherence_config.svh"
module coherence_tb;
  import uvm_pkg::*;
  import coherence_pkg::*;
  logic clock = 0;
  always #5 clock = ~clock;
  coherence_control_if control (clock);
  stream_if #(`COH_REQ_WIDTH) source_if (
      clock,
      control.reset
  );
  stream_if #(`COH_RSP_WIDTH) rx_rsp (
      clock,
      control.reset
  );
  stream_if #(`COH_DAT_WIDTH) rx_dat (
      clock,
      control.reset
  );
  stream_if #(`COH_TXRSP_WIDTH) tx_rsp (
      clock,
      control.reset
  );
  stream_if #(`COH_TXDAT_WIDTH) sink_if (
      clock,
      control.reset
  );
  stream_if #(`COH_SNP_CHANNEL_WIDTH) tx_snp (
      clock,
      control.reset
  );
  stream_if #(`COH_MEMREQ_WIDTH) mem_req (
      clock,
      control.reset
  );
  stream_if #(`COH_MEMRESP_WIDTH) mem_resp (
      clock,
      control.reset
  );
  Coherence dut (
      .clock(clock),
      .reset(control.reset),
      .io_outstanding(control.outstanding),
      `include "coherence_ports.svh"
  );
  initial begin
    uvm_config_db#(virtual coherence_control_if)::set(null, "uvm_test_top*", "control", control);
    uvm_config_db#(virtual stream_if #(`COH_REQ_WIDTH))::set(null, "uvm_test_top*", "req",
                                                             source_if);
    uvm_config_db#(virtual stream_if #(`COH_RSP_WIDTH))::set(null, "uvm_test_top*", "rx_rsp",
                                                             rx_rsp);
    uvm_config_db#(virtual stream_if #(`COH_DAT_WIDTH))::set(null, "uvm_test_top*", "rx_dat",
                                                             rx_dat);
    uvm_config_db#(virtual stream_if #(`COH_TXRSP_WIDTH))::set(null, "uvm_test_top*", "tx_rsp",
                                                               tx_rsp);
    uvm_config_db#(virtual stream_if #(`COH_TXDAT_WIDTH))::set(null, "uvm_test_top*", "tx_dat",
                                                               sink_if);
    uvm_config_db#(virtual stream_if #(`COH_SNP_CHANNEL_WIDTH))::set(null, "uvm_test_top*", "snp",
                                                                     tx_snp);
    uvm_config_db#(virtual stream_if #(`COH_MEMREQ_WIDTH))::set(null, "uvm_test_top*", "mem_req",
                                                                mem_req);
    uvm_config_db#(virtual stream_if #(`COH_MEMRESP_WIDTH))::set(null, "uvm_test_top*", "mem_resp",
                                                                 mem_resp);
    run_test("protocol_test");
  end
endmodule
