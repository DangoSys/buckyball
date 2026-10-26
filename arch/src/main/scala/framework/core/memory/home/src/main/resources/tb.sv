`include "coherence_config.svh"
module home_tb;
  import uvm_pkg::*;
  import home_pkg::*;
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
  stream_if #(`COH_REQ_WIDTH) observedReq (
      clock,
      control.reset
  );
  stream_if #(`COH_RSP_WIDTH) observedRxRsp (
      clock,
      control.reset
  );
  stream_if #(`COH_DAT_WIDTH) observedRxDat (
      clock,
      control.reset
  );
  logic [`COH_AGENTS-1:0] req_valid, req_ready;
  logic [`COH_REQ_WIDTH-1:0] req_bits[`COH_AGENTS];
  logic [`COH_AGENTS-1:0] txRsp_valid, txRsp_ready;
  logic [`COH_RSP_WIDTH-1:0] txRsp_bits[`COH_AGENTS];
  logic [`COH_AGENTS-1:0] txDat_valid, txDat_ready;
  logic [`COH_DAT_WIDTH-1:0] txDat_bits[`COH_AGENTS];
  logic [`COH_AGENTS-1:0] rxRsp_valid, rxRsp_ready;
  logic [`COH_TXRSP_WIDTH-1:0] rxRsp_bits[`COH_AGENTS];
  logic [`COH_AGENTS-1:0] rxDat_valid, rxDat_ready;
  logic [`COH_TXDAT_WIDTH-1:0] rxDat_bits[`COH_AGENTS];
  logic [`COH_AGENTS-1:0] snp_valid, snp_ready;
  logic [`COH_SNP_WIDTH-1:0] snp_bits[`COH_AGENTS];
  wire observedReq_valid;
  wire [`COH_REQ_WIDTH-1:0] observedReq_bits;
  wire observedRxRsp_valid;
  wire [`COH_RSP_WIDTH-1:0] observedRxRsp_bits;
  wire observedRxDat_valid;
  wire [`COH_DAT_WIDTH-1:0] observedRxDat_bits;
  wire observedRsp_valid;
  wire [`COH_TXRSP_WIDTH-1:0] observedRsp_bits;
  wire observedDat_valid;
  wire [`COH_TXDAT_WIDTH-1:0] observedDat_bits;
  wire observedSnp_valid;
  wire [`COH_SNP_CHANNEL_WIDTH-1:0] observedSnp_bits;
  assign observedReq.valid = observedReq_valid;
  assign observedReq.bits = observedReq_bits;
  assign observedReq.ready = 1;
  assign observedRxRsp.valid = observedRxRsp_valid;
  assign observedRxRsp.bits = observedRxRsp_bits;
  assign observedRxRsp.ready = 1;
  assign observedRxDat.valid = observedRxDat_valid;
  assign observedRxDat.bits = observedRxDat_bits;
  assign observedRxDat.ready = 1;
  Home dut (
      .clock(clock),
      .reset(control.reset),
      .io_active(!control.reset),
      .io_blockRequesterRsp(1'b0),
      .io_outstanding(control.outstanding),
      `include "home_ports.svh"
  );
  for (genvar i = 0; i < `COH_AGENTS; i++) begin
    assign req_valid[i] = source_if.valid && source_if.bits[`CF(REQ, SRCID)] == i + 1;
    assign req_bits[i] = source_if.bits;
    assign txRsp_valid[i] = rx_rsp.valid && rx_rsp.bits[`CF(RSP, SRCID)] == i + 1;
    assign txRsp_bits[i] = rx_rsp.bits;
    assign txDat_valid[i] = rx_dat.valid && rx_dat.bits[`CF(DAT, SRCID)] == i + 1;
    assign txDat_bits[i] = rx_dat.bits;
  end
  always_comb begin
    source_if.ready = 0;
    rx_rsp.ready = 0;
    rx_dat.ready = 0;
    for (int i = 0; i < `COH_AGENTS; i++) begin
      source_if.ready |= source_if.bits[`CF(REQ, SRCID)] == i + 1 && req_ready[i];
      rx_rsp.ready |= rx_rsp.bits[`CF(RSP, SRCID)] == i + 1 && txRsp_ready[i];
      rx_dat.ready |= rx_dat.bits[`CF(DAT, SRCID)] == i + 1 && txDat_ready[i];
    end
  end
  int selected_rxRsp = -1;
  always @(posedge clock) if (control.reset || (tx_rsp.valid && tx_rsp.ready)) selected_rxRsp <= -1;
  always @(negedge clock)
    if (!control.reset && selected_rxRsp < 0)
      for (int i = 0; i < `COH_AGENTS; i++)
        if (selected_rxRsp < 0 && rxRsp_valid[i]) selected_rxRsp = i;
  always_comb begin
    tx_rsp.valid = 0;
    tx_rsp.bits  = '0;
    rxRsp_ready  = '0;
    if (selected_rxRsp >= 0) begin
      tx_rsp.valid = rxRsp_valid[selected_rxRsp];
      tx_rsp.bits = rxRsp_bits[selected_rxRsp];
      rxRsp_ready[selected_rxRsp] = tx_rsp.ready;
    end
  end
  int selected_rxDat = -1;
  always @(posedge clock)
    if (control.reset || (sink_if.valid && sink_if.ready))
      selected_rxDat <= -1;
  always @(negedge clock)
    if (!control.reset && selected_rxDat < 0)
      for (int i = 0; i < `COH_AGENTS; i++)
        if (selected_rxDat < 0 && rxDat_valid[i]) selected_rxDat = i;
  always_comb begin
    sink_if.valid = 0;
    sink_if.bits  = '0;
    rxDat_ready   = '0;
    if (selected_rxDat >= 0) begin
      sink_if.valid = rxDat_valid[selected_rxDat];
      sink_if.bits = rxDat_bits[selected_rxDat];
      rxDat_ready[selected_rxDat] = sink_if.ready;
    end
  end
  int selected_snp = -1;
  always @(posedge clock) if (control.reset || (tx_snp.valid && tx_snp.ready)) selected_snp <= -1;
  always @(negedge clock)
    if (!control.reset && selected_snp < 0)
      for (int i = 0; i < `COH_AGENTS; i++) if (selected_snp < 0 && snp_valid[i]) selected_snp = i;
  always_comb begin
    tx_snp.valid = 0;
    tx_snp.bits = '0;
    snp_ready = '0;
    if (selected_snp >= 0) begin
      tx_snp.valid = snp_valid[selected_snp];
      tx_snp.bits = {`COH_SNP_TARGET_WIDTH'(selected_snp + 1), snp_bits[selected_snp]};
      snp_ready[selected_snp] = tx_snp.ready;
    end
  end
  home_transport_if transport (
      clock,
      control.reset
  );
  assign transport.expected_valid[0] = source_if.valid && source_if.ready;
  assign transport.expected_bits[0] = `COH_DAT_WIDTH'(source_if.bits);
  assign transport.expected_node[0] = source_if.bits[`CF(REQ, SRCID)];
  assign transport.actual_valid[0] = observedReq.valid;
  assign transport.actual_bits[0] = `COH_DAT_WIDTH'(observedReq.bits);
  assign transport.actual_node[0] = observedReq.bits[`CF(REQ, SRCID)];
  assign transport.expected_valid[1] = rx_rsp.valid && rx_rsp.ready;
  assign transport.expected_bits[1] = `COH_DAT_WIDTH'(rx_rsp.bits);
  assign transport.expected_node[1] = rx_rsp.bits[`CF(RSP, SRCID)];
  assign transport.actual_valid[1] = observedRxRsp.valid;
  assign transport.actual_bits[1] = `COH_DAT_WIDTH'(observedRxRsp.bits);
  assign transport.actual_node[1] = observedRxRsp.bits[`CF(RSP, SRCID)];
  assign transport.expected_valid[2] = rx_dat.valid && rx_dat.ready;
  assign transport.expected_bits[2] = `COH_DAT_WIDTH'(rx_dat.bits);
  assign transport.expected_node[2] = rx_dat.bits[`CF(DAT, SRCID)];
  assign transport.actual_valid[2] = observedRxDat.valid;
  assign transport.actual_bits[2] = `COH_DAT_WIDTH'(observedRxDat.bits);
  assign transport.actual_node[2] = observedRxDat.bits[`CF(DAT, SRCID)];
  assign transport.expected_valid[3] = observedRsp_valid;
  assign transport.expected_bits[3] = `COH_DAT_WIDTH'(observedRsp_bits);
  assign transport.expected_node[3] = observedRsp_bits[`CF(TXRSP, TGTID)];
  assign transport.actual_valid[3] = tx_rsp.valid && tx_rsp.ready;
  assign transport.actual_bits[3] = `COH_DAT_WIDTH'(tx_rsp.bits);
  assign transport.actual_node[3] = tx_rsp.bits[`CF(TXRSP, TGTID)];
  assign transport.expected_valid[4] = observedDat_valid;
  assign transport.expected_bits[4] = `COH_DAT_WIDTH'(observedDat_bits);
  assign transport.expected_node[4] = observedDat_bits[`CF(TXDAT, TGTID)];
  assign transport.actual_valid[4] = sink_if.valid && sink_if.ready;
  assign transport.actual_bits[4] = `COH_DAT_WIDTH'(sink_if.bits);
  assign transport.actual_node[4] = sink_if.bits[`CF(TXDAT, TGTID)];
  assign transport.expected_valid[5] = observedSnp_valid;
  assign transport.expected_bits[5] = `COH_DAT_WIDTH'(observedSnp_bits);
  assign transport.expected_node[5] = observedSnp_bits[`CF(SNP, TARGET)];
  assign transport.actual_valid[5] = tx_snp.valid && tx_snp.ready;
  assign transport.actual_bits[5] = `COH_DAT_WIDTH'(tx_snp.bits);
  assign transport.actual_node[5] = tx_snp.bits[`CF(SNP, TARGET)];
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
    uvm_config_db#(virtual stream_if #(`COH_REQ_WIDTH))::set(null, "uvm_test_top.env.bus.mon",
                                                             "req", observedReq);
    uvm_config_db#(virtual stream_if #(`COH_RSP_WIDTH))::set(null, "uvm_test_top.env.bus.mon",
                                                             "rx_rsp", observedRxRsp);
    uvm_config_db#(virtual stream_if #(`COH_DAT_WIDTH))::set(null, "uvm_test_top.env.bus.mon",
                                                             "rx_dat", observedRxDat);
    uvm_config_db#(virtual home_transport_if)::set(null, "uvm_test_top.transport", "vif",
                                                   transport);
    run_test("home_test");
  end
endmodule
