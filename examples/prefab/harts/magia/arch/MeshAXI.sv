`include "axi/typedef.svh"
`include "axi/assign.svh"

(* buckyball_p2e_native *)
module MagiaMeshAXI (
    input logic clock,
    input logic reset,
    input logic [31:0] boot_address,
    input logic fetch_enable,
    output logic aw_valid,
    input logic aw_ready,
    output logic [5:0] aw_bits_id,
    output logic [31:0] aw_bits_addr,
    output logic [7:0] aw_bits_len,
    output logic [2:0] aw_bits_size,
    output logic [1:0] aw_bits_burst,
    output logic aw_bits_lock,
    output logic [3:0] aw_bits_cache,
    output logic [2:0] aw_bits_prot,
    output logic [3:0] aw_bits_qos,
    output logic w_valid,
    input logic w_ready,
    output logic [127:0] w_bits_data,
    output logic [15:0] w_bits_strb,
    output logic w_bits_last,
    input logic b_valid,
    output logic b_ready,
    input logic [5:0] b_bits_id,
    input logic [1:0] b_bits_resp,
    output logic ar_valid,
    input logic ar_ready,
    output logic [5:0] ar_bits_id,
    output logic [31:0] ar_bits_addr,
    output logic [7:0] ar_bits_len,
    output logic [2:0] ar_bits_size,
    output logic [1:0] ar_bits_burst,
    output logic ar_bits_lock,
    output logic [3:0] ar_bits_cache,
    output logic [2:0] ar_bits_prot,
    output logic [3:0] ar_bits_qos,
    input logic r_valid,
    output logic r_ready,
    input logic [5:0] r_bits_id,
    input logic [127:0] r_bits_data,
    input logic [1:0] r_bits_resp,
    input logic r_bits_last
);
  import magia_pkg::*;
  import magia_tile_pkg::*;
  import magia_noc_pkg::*;

  floo_req_t [3:0] req_to_l2, req_to_mesh;
  floo_rsp_t [3:0] rsp_to_l2, rsp_to_mesh;
  floo_wide_t [3:0] wide_to_l2, wide_to_mesh;
  axi_wide_l2_req_t [7:0] requests;
  axi_wide_l2_rsp_t [7:0] responses;
  logic [31:0] irq[16];
  logic [31:0] control_pc[16];
  logic control_pc_valid[16];
  for (genvar i = 0; i < 16; i++) assign irq[i] = '0;

  MagiaMesh mesh (
      .clk_i(clock),
      .rst_ni(~reset),
      .test_mode_i(1'b0),
      .tile_enable_i(1'b1),
      .scan_cg_en_i(1'b0),
      .boot_addr_i(boot_address),
      .mtvec_addr_i(boot_address),
      .dm_halt_addr_i('0),
      .dm_exception_addr_i('0),
      .mimpid_patch_i('0),
      .mcycle_o(),
      .irq_i(irq),
      .debug_req_i(1'b0),
      .debug_havereset_o(),
      .debug_running_o(),
      .debug_halted_o(),
      .debug_pc_valid_o(control_pc_valid),
      .debug_pc_o(control_pc),
      .fetch_enable_i(fetch_enable),
      .core_sleep_o(),
      .wu_wfe_i(1'b0),
      .l2_noc_req_o(req_to_l2),
      .l2_noc_rsp_i(rsp_to_mesh),
      .l2_noc_req_i(req_to_mesh),
      .l2_noc_rsp_o(rsp_to_l2),
      .l2_noc_wide_i(wide_to_mesh),
      .l2_noc_wide_o(wide_to_l2)
`ifdef VERILATOR,
      .tile_observe_o()
`endif
  );
  MagiaMemory #(
      .NumPorts(4)
  ) links (
      .clk_i(clock),
      .rst_ni(~reset),
      .noc_req_i(req_to_l2),
      .noc_rsp_o(rsp_to_mesh),
      .noc_req_o(req_to_mesh),
      .noc_rsp_i(rsp_to_l2),
      .noc_wide_i(wide_to_l2),
      .noc_wide_o(wide_to_mesh),
      .l2_data_req(requests),
      .l2_data_rsp(responses)
  );

  `AXI_TYPEDEF_ALL(host, logic[31:0], logic[5:0], logic[255:0], logic[31:0], logic[35:0])
  host_req_t  mux_req;
  host_resp_t mux_rsp;
  axi_mux #(
      .SlvAxiIDWidth(3),
      .NoSlvPorts(8),
      .MaxWTrans(8),
      .slv_aw_chan_t(axi_wide_l2_aw_chan_t),
      .mst_aw_chan_t(host_aw_chan_t),
      .w_chan_t(host_w_chan_t),
      .slv_b_chan_t(axi_wide_l2_b_chan_t),
      .mst_b_chan_t(host_b_chan_t),
      .slv_ar_chan_t(axi_wide_l2_ar_chan_t),
      .mst_ar_chan_t(host_ar_chan_t),
      .slv_r_chan_t(axi_wide_l2_r_chan_t),
      .mst_r_chan_t(host_r_chan_t),
      .slv_req_t(axi_wide_l2_req_t),
      .slv_resp_t(axi_wide_l2_rsp_t),
      .mst_req_t(host_req_t),
      .mst_resp_t(host_resp_t)
  ) mux (
      .clk_i(clock),
      .rst_ni(~reset),
      .test_i(1'b0),
      .slv_reqs_i(requests),
      .slv_resps_o(responses),
      .mst_req_o(mux_req),
      .mst_resp_i(mux_rsp)
  );
  AXI_BUS #(
      .AXI_ADDR_WIDTH(32),
      .AXI_DATA_WIDTH(256),
      .AXI_ID_WIDTH  (6),
      .AXI_USER_WIDTH(36)
  )
      atomic_in (), host_bus ();
  `AXI_ASSIGN_FROM_REQ(atomic_in, mux_req)
  `AXI_ASSIGN_TO_RESP(mux_rsp, atomic_in)
  axi_riscv_atomics_wrap #(
      .AXI_ADDR_WIDTH(32),
      .AXI_DATA_WIDTH(256),
      .AXI_ID_WIDTH(6),
      .AXI_USER_WIDTH(36),
      .AXI_MAX_READ_TXNS(8),
      .AXI_MAX_WRITE_TXNS(8),
      .RISCV_WORD_WIDTH(32)
  ) atomics (
      .clk_i(clock),
      .rst_ni(~reset),
      .slv(atomic_in),
      .mst(host_bus)
  );
  `AXI_TYPEDEF_ALL(edge, logic[31:0], logic[5:0], logic[127:0], logic[15:0], logic[35:0])
  host_req_t  wide_req;
  host_resp_t wide_rsp;
  edge_req_t  edge_req;
  edge_resp_t edge_rsp;
  AXI_BUS #(
      .AXI_ADDR_WIDTH(32),
      .AXI_DATA_WIDTH(128),
      .AXI_ID_WIDTH  (6),
      .AXI_USER_WIDTH(36)
  ) out_bus ();
  `AXI_ASSIGN_TO_REQ(wide_req, host_bus)
  `AXI_ASSIGN_FROM_RESP(host_bus, wide_rsp)
  // A tile's DMA moves whole beats of the wide bus and marks the bytes of a
  // transfer with strobes; the low bits of its address are not the first
  // byte (they read 3). The width converter takes the address for the first
  // byte and drops the lanes below it, so a whole-beat transfer goes in at
  // the address of its beat.
  host_req_t beat_req;
  always_comb begin
    beat_req = wide_req;
    if (wide_req.aw.size == 3'd5) beat_req.aw.addr[4:0] = '0;
    if (wide_req.ar.size == 3'd5) beat_req.ar.addr[4:0] = '0;
  end
  `AXI_ASSIGN_FROM_REQ(out_bus, edge_req)
  `AXI_ASSIGN_TO_RESP(edge_rsp, out_bus)
  axi_dw_converter #(
      .AxiMaxReads(8),
      .AxiSlvPortDataWidth(256),
      .AxiMstPortDataWidth(128),
      .AxiAddrWidth(32),
      .AxiIdWidth(6),
      .aw_chan_t(host_aw_chan_t),
      .ar_chan_t(host_ar_chan_t),
      .b_chan_t(host_b_chan_t),
      .slv_w_chan_t(host_w_chan_t),
      .mst_w_chan_t(edge_w_chan_t),
      .slv_r_chan_t(host_r_chan_t),
      .mst_r_chan_t(edge_r_chan_t),
      .axi_slv_req_t(host_req_t),
      .axi_slv_resp_t(host_resp_t),
      .axi_mst_req_t(edge_req_t),
      .axi_mst_resp_t(edge_resp_t)
  ) host_width (
      .clk_i(clock),
      .rst_ni(~reset),
      .slv_req_i(beat_req),
      .slv_resp_o(wide_rsp),
      .mst_req_o(edge_req),
      .mst_resp_i(edge_rsp)
  );
  assign aw_valid = out_bus.aw_valid;
  assign out_bus.aw_ready = aw_ready;
  assign aw_bits_id = out_bus.aw_id;
  assign aw_bits_addr = out_bus.aw_addr;
  assign aw_bits_len = out_bus.aw_len;
  assign aw_bits_size = out_bus.aw_size;
  assign aw_bits_burst = out_bus.aw_burst;
  assign aw_bits_lock = out_bus.aw_lock;
  assign aw_bits_cache = out_bus.aw_cache;
  assign aw_bits_prot = out_bus.aw_prot;
  assign aw_bits_qos = out_bus.aw_qos;
  assign w_valid = out_bus.w_valid;
  assign out_bus.w_ready = w_ready;
  assign w_bits_data = out_bus.w_data;
  assign w_bits_strb = out_bus.w_strb;
  assign w_bits_last = out_bus.w_last;
  assign out_bus.b_valid = b_valid;
  assign b_ready = out_bus.b_ready;
  assign out_bus.b_id = b_bits_id;
  assign out_bus.b_resp = b_bits_resp;
  assign ar_valid = out_bus.ar_valid;
  assign out_bus.ar_ready = ar_ready;
  assign ar_bits_id = out_bus.ar_id;
  assign ar_bits_addr = out_bus.ar_addr;
  assign ar_bits_len = out_bus.ar_len;
  assign ar_bits_size = out_bus.ar_size;
  assign ar_bits_burst = out_bus.ar_burst;
  assign ar_bits_lock = out_bus.ar_lock;
  assign ar_bits_cache = out_bus.ar_cache;
  assign ar_bits_prot = out_bus.ar_prot;
  assign ar_bits_qos = out_bus.ar_qos;
  assign out_bus.r_valid = r_valid;
  assign r_ready = out_bus.r_ready;
  assign out_bus.r_id = r_bits_id;
  assign out_bus.r_data = r_bits_data;
  assign out_bus.r_resp = r_bits_resp;
  assign out_bus.r_last = r_bits_last;
  assign out_bus.b_user = '0;
  assign out_bus.r_user = '0;
endmodule
