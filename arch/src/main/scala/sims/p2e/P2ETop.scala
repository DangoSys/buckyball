package sims.p2e

import chisel3._
import chisel3.experimental.{attach, Analog}
import chisel3.util.HasBlackBoxInline

class P2ETopIO extends Bundle {
  val user_clk = Input(Clock())
  val sys_rstn = Input(Bool())

  val c0_sys_clk_p = Input(Bool())
  val c0_sys_clk_n = Input(Bool())

  val c0_ddr4_act_n       = Output(Bool())
  val c0_ddr4_adr         = Output(UInt(17.W))
  val c0_ddr4_ba          = Output(UInt(2.W))
  val c0_ddr4_bg          = Output(UInt(2.W))
  val c0_ddr4_cke         = Output(UInt(2.W))
  val c0_ddr4_odt         = Output(UInt(2.W))
  val c0_ddr4_cs_n        = Output(UInt(2.W))
  val c0_ddr4_ck_t        = Output(UInt(2.W))
  val c0_ddr4_ck_c        = Output(UInt(2.W))
  val c0_ddr4_reset_n     = Output(Bool())
  val c0_ddr4_dm_dbi_n    = Analog(8.W)
  val c0_ddr4_dq          = Analog(64.W)
  val c0_ddr4_dqs_c       = Analog(8.W)
  val c0_ddr4_dqs_t       = Analog(8.W)
  val c0_ddr4_ui_clk      = Output(Clock())
  val init_calib_complete = Output(Bool())

  val ddr4_en_vtt    = Output(Bool())
  val ddr4_en_vddq   = Output(Bool())
  val ddr4_en_vcc2v5 = Output(Bool())
  val power_good     = Input(Bool())
}

class P2ETopBlackBox extends BlackBox with HasBlackBoxInline {
  val io = IO(new P2ETopIO)

  setInline(
    "P2ETopBlackBox.v",
    """
      |module p2e_mmio_zero_slave #(
      |  parameter ADDR_BITS = 32,
      |  parameter DATA_BITS = 64,
      |  parameter ID_BITS   = 4,
      |  parameter STRB_BITS = DATA_BITS / 8
      |)(
      |  input                    clock,
      |  input                    reset,
      |  output                   aw_ready,
      |  input                    aw_valid,
      |  input  [ID_BITS-1:0]     aw_id,
      |  input  [ADDR_BITS-1:0]   aw_addr,
      |  input  [7:0]             aw_len,
      |  input  [2:0]             aw_size,
      |  input  [1:0]             aw_burst,
      |  output                   w_ready,
      |  input                    w_valid,
      |  input  [DATA_BITS-1:0]   w_data,
      |  input  [STRB_BITS-1:0]   w_strb,
      |  input                    w_last,
      |  input                    b_ready,
      |  output reg               b_valid,
      |  output reg [ID_BITS-1:0] b_id,
      |  output [1:0]             b_resp,
      |  output                   ar_ready,
      |  input                    ar_valid,
      |  input  [ID_BITS-1:0]     ar_id,
      |  input  [ADDR_BITS-1:0]   ar_addr,
      |  input  [7:0]             ar_len,
      |  input  [2:0]             ar_size,
      |  input  [1:0]             ar_burst,
      |  input                    r_ready,
      |  output reg               r_valid,
      |  output reg [ID_BITS-1:0] r_id,
      |  output [DATA_BITS-1:0]   r_data,
      |  output [1:0]             r_resp,
      |  output                   r_last
      |);
      |  assign aw_ready = !b_valid;
      |  assign w_ready  = !b_valid;
      |  assign b_resp   = 2'b00;
      |
      |  assign ar_ready = !r_valid;
      |  assign r_data   = {DATA_BITS{1'b0}};
      |  assign r_resp   = 2'b00;
      |  assign r_last   = 1'b1;
      |
      |  always @(posedge clock) begin
      |    if (reset) begin
      |      b_valid <= 1'b0;
      |      b_id    <= {ID_BITS{1'b0}};
      |      r_valid <= 1'b0;
      |      r_id    <= {ID_BITS{1'b0}};
      |    end else begin
      |      if (b_valid && b_ready) begin
      |        b_valid <= 1'b0;
      |      end
      |      if (!b_valid && aw_valid && w_valid) begin
      |        b_valid <= 1'b1;
      |        b_id    <= aw_id;
      |      end
      |
      |      if (r_valid && r_ready) begin
      |        r_valid <= 1'b0;
      |      end
      |      if (!r_valid && ar_valid) begin
      |        r_valid <= 1'b1;
      |        r_id    <= ar_id;
      |      end
      |    end
      |  end
      |endmodule
      |
      |module P2ETopBlackBox(
      |  input         user_clk,
      |  input         sys_rstn,
      |  input         c0_sys_clk_p,
      |  input         c0_sys_clk_n,
      |  output        c0_ddr4_act_n,
      |  output [16:0] c0_ddr4_adr,
      |  output [1:0]  c0_ddr4_ba,
      |  output [1:0]  c0_ddr4_bg,
      |  output [1:0]  c0_ddr4_cke,
      |  output [1:0]  c0_ddr4_odt,
      |  output [1:0]  c0_ddr4_cs_n,
      |  output [1:0]  c0_ddr4_ck_t,
      |  output [1:0]  c0_ddr4_ck_c,
      |  output        c0_ddr4_reset_n,
      |  inout  [7:0]  c0_ddr4_dm_dbi_n,
      |  inout  [63:0] c0_ddr4_dq,
      |  inout  [7:0]  c0_ddr4_dqs_c,
      |  inout  [7:0]  c0_ddr4_dqs_t,
      |  output        c0_ddr4_ui_clk,
      |  output        init_calib_complete,
      |  output        ddr4_en_vtt,
      |  output        ddr4_en_vddq,
      |  output        ddr4_en_vcc2v5,
      |  input         power_good
      |);
      |  localparam MEM_ADDR_BITS = 64;
      |  localparam MEM_DATA_BITS = 256;
      |  localparam MEM_STRB_BITS = 32;
      |  localparam MEM_ID_BITS   = 11;
      |  localparam MMIO_ADDR_BITS = 32;
      |  localparam MMIO_DATA_BITS = 64;
      |  localparam MMIO_STRB_BITS = 8;
      |  localparam MMIO_ID_BITS   = 4;
      |
      |  wire c0_init_calib_complete;
      |  wire init_calib_complete_unused;  // Unused output from DDR4 controller
      |  wire soc_reset = !sys_rstn || !c0_init_calib_complete;
      |  assign init_calib_complete = c0_init_calib_complete;
      |
      |  wire [MEM_ID_BITS-1:0]     mem_awid;
      |  wire [MEM_ADDR_BITS-1:0]   mem_awaddr;
      |  wire [7:0]                 mem_awlen;
      |  wire [2:0]                 mem_awsize;
      |  wire [1:0]                 mem_awburst;
      |  wire                       mem_awvalid;
      |  wire                       mem_awready;
      |  wire [MEM_DATA_BITS-1:0]   mem_wdata;
      |  wire [MEM_STRB_BITS-1:0]   mem_wstrb;
      |  wire                       mem_wlast;
      |  wire                       mem_wvalid;
      |  wire                       mem_wready;
      |  wire [MEM_ID_BITS-1:0]     mem_bid;
      |  wire [1:0]                 mem_bresp;
      |  wire                       mem_bvalid;
      |  wire                       mem_bready;
      |  wire [MEM_ID_BITS-1:0]     mem_arid;
      |  wire [MEM_ADDR_BITS-1:0]   mem_araddr;
      |  wire [7:0]                 mem_arlen;
      |  wire [2:0]                 mem_arsize;
      |  wire [1:0]                 mem_arburst;
      |  wire                       mem_arvalid;
      |  wire                       mem_arready;
      |  wire [MEM_ID_BITS-1:0]     mem_rid;
      |  wire [MEM_DATA_BITS-1:0]   mem_rdata;
      |  wire [1:0]                 mem_rresp;
      |  wire                       mem_rlast;
      |  wire                       mem_rvalid;
      |  wire                       mem_rready;
      |
      |  wire [MMIO_ID_BITS-1:0]    mmio_awid;
      |  wire [MMIO_ADDR_BITS-1:0]  mmio_awaddr;
      |  wire [7:0]                 mmio_awlen;
      |  wire [2:0]                 mmio_awsize;
      |  wire [1:0]                 mmio_awburst;
      |  wire                       mmio_awvalid;
      |  wire                       mmio_awready;
      |  wire [MMIO_DATA_BITS-1:0]  mmio_wdata;
      |  wire [MMIO_STRB_BITS-1:0]  mmio_wstrb;
      |  wire                       mmio_wlast;
      |  wire                       mmio_wvalid;
      |  wire                       mmio_wready;
      |  wire [MMIO_ID_BITS-1:0]    mmio_bid;
      |  wire [1:0]                 mmio_bresp;
      |  wire                       mmio_bvalid;
      |  wire                       mmio_bready;
      |  wire [MMIO_ID_BITS-1:0]    mmio_arid;
      |  wire [MMIO_ADDR_BITS-1:0]  mmio_araddr;
      |  wire [7:0]                 mmio_arlen;
      |  wire [2:0]                 mmio_arsize;
      |  wire [1:0]                 mmio_arburst;
      |  wire                       mmio_arvalid;
      |  wire                       mmio_arready;
      |  wire [MMIO_ID_BITS-1:0]    mmio_rid;
      |  wire [MMIO_DATA_BITS-1:0]  mmio_rdata;
      |  wire [1:0]                 mmio_rresp;
      |  wire                       mmio_rlast;
      |  wire                       mmio_rvalid;
      |  wire                       mmio_rready;
      |
      |  DigitalTop soc (
      |    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock (user_clk),
      |    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_reset (soc_reset),
      |    // Debug ports removed - WithNoDebug config doesn't generate these ports
      |    // .resetctrl_hartIsInReset_0 (1'b0),
      |    // .debug_clock               (user_clk),
      |    // .debug_reset               (soc_reset),
      |    // .debug_systemjtag_reset    (soc_reset),
      |    // .debug_systemjtag_jtag_TCK (1'b0),
      |    // .debug_systemjtag_jtag_TMS (1'b1),
      |    // .debug_systemjtag_jtag_TDI (1'b1),
      |    // .debug_dmactiveAck         (1'b0),
      |    .mem_axi4_0_aw_ready       (mem_awready),
      |    .mem_axi4_0_aw_valid       (mem_awvalid),
      |    .mem_axi4_0_aw_bits_id     (mem_awid),
      |    .mem_axi4_0_aw_bits_addr   (mem_awaddr),
      |    .mem_axi4_0_aw_bits_len    (mem_awlen),
      |    .mem_axi4_0_aw_bits_size   (mem_awsize),
      |    .mem_axi4_0_aw_bits_burst  (mem_awburst),
      |    .mem_axi4_0_w_ready        (mem_wready),
      |    .mem_axi4_0_w_valid        (mem_wvalid),
      |    .mem_axi4_0_w_bits_data    (mem_wdata),
      |    .mem_axi4_0_w_bits_strb    (mem_wstrb),
      |    .mem_axi4_0_w_bits_last    (mem_wlast),
      |    .mem_axi4_0_b_valid        (mem_bvalid),
      |    .mem_axi4_0_b_ready        (mem_bready),
      |    .mem_axi4_0_b_bits_id      (mem_bid),
      |    .mem_axi4_0_b_bits_resp    (mem_bresp),
      |    .mem_axi4_0_ar_ready       (mem_arready),
      |    .mem_axi4_0_ar_valid       (mem_arvalid),
      |    .mem_axi4_0_ar_bits_id     (mem_arid),
      |    .mem_axi4_0_ar_bits_addr   (mem_araddr),
      |    .mem_axi4_0_ar_bits_len    (mem_arlen),
      |    .mem_axi4_0_ar_bits_size   (mem_arsize),
      |    .mem_axi4_0_ar_bits_burst  (mem_arburst),
      |    .mem_axi4_0_r_valid        (mem_rvalid),
      |    .mem_axi4_0_r_ready        (mem_rready),
      |    .mem_axi4_0_r_bits_id      (mem_rid),
      |    .mem_axi4_0_r_bits_data    (mem_rdata),
      |    .mem_axi4_0_r_bits_resp    (mem_rresp),
      |    .mem_axi4_0_r_bits_last    (mem_rlast),
      |    .mmio_axi4_0_aw_ready      (mmio_awready),
      |    .mmio_axi4_0_aw_valid      (mmio_awvalid),
      |    .mmio_axi4_0_aw_bits_id    (mmio_awid),
      |    .mmio_axi4_0_aw_bits_addr  (mmio_awaddr),
      |    .mmio_axi4_0_aw_bits_len   (mmio_awlen),
      |    .mmio_axi4_0_aw_bits_size  (mmio_awsize),
      |    .mmio_axi4_0_aw_bits_burst (mmio_awburst),
      |    .mmio_axi4_0_w_ready       (mmio_wready),
      |    .mmio_axi4_0_w_valid       (mmio_wvalid),
      |    .mmio_axi4_0_w_bits_data   (mmio_wdata),
      |    .mmio_axi4_0_w_bits_strb   (mmio_wstrb),
      |    .mmio_axi4_0_w_bits_last   (mmio_wlast),
      |    .mmio_axi4_0_b_valid       (mmio_bvalid),
      |    .mmio_axi4_0_b_ready       (mmio_bready),
      |    .mmio_axi4_0_b_bits_id     (mmio_bid),
      |    .mmio_axi4_0_b_bits_resp   (mmio_bresp),
      |    .mmio_axi4_0_ar_ready      (mmio_arready),
      |    .mmio_axi4_0_ar_valid      (mmio_arvalid),
      |    .mmio_axi4_0_ar_bits_id    (mmio_arid),
      |    .mmio_axi4_0_ar_bits_addr  (mmio_araddr),
      |    .mmio_axi4_0_ar_bits_len   (mmio_arlen),
      |    .mmio_axi4_0_ar_bits_size  (mmio_arsize),
      |    .mmio_axi4_0_ar_bits_burst (mmio_arburst),
      |    .mmio_axi4_0_r_valid       (mmio_rvalid),
      |    .mmio_axi4_0_r_ready       (mmio_rready),
      |    .mmio_axi4_0_r_bits_id     (mmio_rid),
      |    .mmio_axi4_0_r_bits_data   (mmio_rdata),
      |    .mmio_axi4_0_r_bits_resp   (mmio_rresp),
      |    .mmio_axi4_0_r_bits_last   (mmio_rlast),
      |    .serial_tl_0_in_valid      (1'b0),
      |    .serial_tl_0_in_bits_phit  (32'h0),
      |    .serial_tl_0_out_ready     (1'b0),
      |    .serial_tl_0_clock_in      (1'b0),
      |    .custom_boot               (1'b0)
      |  );
      |
      |  p2e_mmio_zero_slave #(
      |    .ADDR_BITS(MMIO_ADDR_BITS),
      |    .DATA_BITS(MMIO_DATA_BITS),
      |    .ID_BITS(MMIO_ID_BITS)
      |  ) mmio_stub (
      |    .clock    (user_clk),
      |    .reset    (soc_reset),
      |    .aw_ready (mmio_awready),
      |    .aw_valid (mmio_awvalid),
      |    .aw_id    (mmio_awid),
      |    .aw_addr  (mmio_awaddr),
      |    .aw_len   (mmio_awlen),
      |    .aw_size  (mmio_awsize),
      |    .aw_burst (mmio_awburst),
      |    .w_ready  (mmio_wready),
      |    .w_valid  (mmio_wvalid),
      |    .w_data   (mmio_wdata),
      |    .w_strb   (mmio_wstrb),
      |    .w_last   (mmio_wlast),
      |    .b_ready  (mmio_bready),
      |    .b_valid  (mmio_bvalid),
      |    .b_id     (mmio_bid),
      |    .b_resp   (mmio_bresp),
      |    .ar_ready (mmio_arready),
      |    .ar_valid (mmio_arvalid),
      |    .ar_id    (mmio_arid),
      |    .ar_addr  (mmio_araddr),
      |    .ar_len   (mmio_arlen),
      |    .ar_size  (mmio_arsize),
      |    .ar_burst (mmio_arburst),
      |    .r_ready  (mmio_rready),
      |    .r_valid  (mmio_rvalid),
      |    .r_id     (mmio_rid),
      |    .r_data   (mmio_rdata),
      |    .r_resp   (mmio_rresp),
      |    .r_last   (mmio_rlast)
      |  );
      |
      |  xepic_ddr4_dc1 ddr (
      |    .sys_rstn                 (sys_rstn),
      |    .c0_sys_clk_p             (c0_sys_clk_p),
      |    .c0_sys_clk_n             (c0_sys_clk_n),
      |    .c0_ddr4_act_n            (c0_ddr4_act_n),
      |    .c0_ddr4_adr              (c0_ddr4_adr),
      |    .c0_ddr4_ba               (c0_ddr4_ba),
      |    .c0_ddr4_bg               (c0_ddr4_bg),
      |    .c0_ddr4_cke              (c0_ddr4_cke),
      |    .c0_ddr4_odt              (c0_ddr4_odt),
      |    .c0_ddr4_cs_n             (c0_ddr4_cs_n),
      |    .c0_ddr4_ck_t             (c0_ddr4_ck_t),
      |    .c0_ddr4_ck_c             (c0_ddr4_ck_c),
      |    .c0_ddr4_reset_n          (c0_ddr4_reset_n),
      |    .c0_ddr4_dm_dbi_n         (c0_ddr4_dm_dbi_n),
      |    .c0_ddr4_dq               (c0_ddr4_dq),
      |    .c0_ddr4_dqs_c            (c0_ddr4_dqs_c),
      |    .c0_ddr4_dqs_t            (c0_ddr4_dqs_t),
      |    .gclk_100m                (1'b0),
      |    .ddr4_en_vtt_bbox         (ddr4_en_vtt),
      |    .ddr4_en_vddq_bbox        (ddr4_en_vddq),
      |    .ddr4_en_vcc2v5_bbox      (ddr4_en_vcc2v5),
      |    .power_good_bbox          (power_good),
      |    .init_start               (1'b1),
      |    .init_cfg                 (1'b0),
      |    .init_busy                (),
      |    .init_calib_complete      (init_calib_complete_unused),
      |    .c0_init_calib_complete   (c0_init_calib_complete),
      |    .axi_clk                  (user_clk),
      |    .s0_ddr4_s_axi_awid       (mem_awid),
      |    .s0_ddr4_s_axi_awaddr     (mem_awaddr),
      |    .s0_ddr4_s_axi_awlen      (mem_awlen),
      |    .s0_ddr4_s_axi_awsize     (mem_awsize),
      |    .s0_ddr4_s_axi_awburst    (mem_awburst),
      |    .s0_ddr4_s_axi_awlock     (1'b0),
      |    .s0_ddr4_s_axi_awcache    (4'b0011),
      |    .s0_ddr4_s_axi_awprot     (3'b000),
      |    .s0_ddr4_s_axi_awqos      (4'b0000),
      |    .s0_ddr4_s_axi_awvalid    (mem_awvalid),
      |    .s0_ddr4_s_axi_awready    (mem_awready),
      |    .s0_ddr4_s_axi_wdata      (mem_wdata),
      |    .s0_ddr4_s_axi_wstrb      (mem_wstrb),
      |    .s0_ddr4_s_axi_wlast      (mem_wlast),
      |    .s0_ddr4_s_axi_wvalid     (mem_wvalid),
      |    .s0_ddr4_s_axi_wready     (mem_wready),
      |    .s0_ddr4_s_axi_bready     (mem_bready),
      |    .s0_ddr4_s_axi_bid        (mem_bid),
      |    .s0_ddr4_s_axi_bresp      (mem_bresp),
      |    .s0_ddr4_s_axi_bvalid     (mem_bvalid),
      |    .s0_ddr4_s_axi_arid       (mem_arid),
      |    .s0_ddr4_s_axi_araddr     (mem_araddr),
      |    .s0_ddr4_s_axi_arlen      (mem_arlen),
      |    .s0_ddr4_s_axi_arsize     (mem_arsize),
      |    .s0_ddr4_s_axi_arburst    (mem_arburst),
      |    .s0_ddr4_s_axi_arlock     (1'b0),
      |    .s0_ddr4_s_axi_arcache    (4'b0011),
      |    .s0_ddr4_s_axi_arprot     (3'b000),
      |    .s0_ddr4_s_axi_arqos      (4'b0000),
      |    .s0_ddr4_s_axi_arvalid    (mem_arvalid),
      |    .s0_ddr4_s_axi_arready    (mem_arready),
      |    .s0_ddr4_s_axi_rready     (mem_rready),
      |    .s0_ddr4_s_axi_rid        (mem_rid),
      |    .s0_ddr4_s_axi_rdata      (mem_rdata),
      |    .s0_ddr4_s_axi_rresp      (mem_rresp),
      |    .s0_ddr4_s_axi_rlast      (mem_rlast),
      |    .s0_ddr4_s_axi_rvalid     (mem_rvalid),
      |    .c0_ddr4_ui_clk           (c0_ddr4_ui_clk),
      |    .s1_ddr4_s_axi_awid       (4'b0),
      |    .s1_ddr4_s_axi_awaddr     (64'b0),
      |    .s1_ddr4_s_axi_awlen      (8'b0),
      |    .s1_ddr4_s_axi_awsize     (3'b0),
      |    .s1_ddr4_s_axi_awburst    (2'b0),
      |    .s1_ddr4_s_axi_awlock     (1'b0),
      |    .s1_ddr4_s_axi_awcache    (4'b0),
      |    .s1_ddr4_s_axi_awprot     (3'b0),
      |    .s1_ddr4_s_axi_awqos      (4'b0),
      |    .s1_ddr4_s_axi_awvalid    (1'b0),
      |    .s1_ddr4_s_axi_awready    (),
      |    .s1_ddr4_s_axi_wdata      (256'b0),
      |    .s1_ddr4_s_axi_wstrb      (32'b0),
      |    .s1_ddr4_s_axi_wlast      (1'b0),
      |    .s1_ddr4_s_axi_wvalid     (1'b0),
      |    .s1_ddr4_s_axi_wready     (),
      |    .s1_ddr4_s_axi_bready     (1'b1),
      |    .s1_ddr4_s_axi_bid        (),
      |    .s1_ddr4_s_axi_bresp      (),
      |    .s1_ddr4_s_axi_bvalid     (),
      |    .s1_ddr4_s_axi_arid       (4'b0),
      |    .s1_ddr4_s_axi_araddr     (64'b0),
      |    .s1_ddr4_s_axi_arlen      (8'b0),
      |    .s1_ddr4_s_axi_arsize     (3'b0),
      |    .s1_ddr4_s_axi_arburst    (2'b0),
      |    .s1_ddr4_s_axi_arlock     (1'b0),
      |    .s1_ddr4_s_axi_arcache    (4'b0),
      |    .s1_ddr4_s_axi_arprot     (3'b0),
      |    .s1_ddr4_s_axi_arqos      (4'b0),
      |    .s1_ddr4_s_axi_arvalid    (1'b0),
      |    .s1_ddr4_s_axi_arready    (),
      |    .s1_ddr4_s_axi_rready     (1'b1),
      |    .s1_ddr4_s_axi_rid        (),
      |    .s1_ddr4_s_axi_rdata      (),
      |    .s1_ddr4_s_axi_rresp      (),
      |    .s1_ddr4_s_axi_rlast      (),
      |    .s1_ddr4_s_axi_rvalid     ()
      |  );
      |endmodule
    """.stripMargin
  )
}

class P2ETop extends RawModule {
  val io = IO(new P2ETopIO)

  val top = Module(new P2ETopBlackBox)
  top.io.user_clk        := io.user_clk
  top.io.sys_rstn        := io.sys_rstn
  top.io.c0_sys_clk_p    := io.c0_sys_clk_p
  top.io.c0_sys_clk_n    := io.c0_sys_clk_n
  io.c0_ddr4_act_n       := top.io.c0_ddr4_act_n
  io.c0_ddr4_adr         := top.io.c0_ddr4_adr
  io.c0_ddr4_ba          := top.io.c0_ddr4_ba
  io.c0_ddr4_bg          := top.io.c0_ddr4_bg
  io.c0_ddr4_cke         := top.io.c0_ddr4_cke
  io.c0_ddr4_odt         := top.io.c0_ddr4_odt
  io.c0_ddr4_cs_n        := top.io.c0_ddr4_cs_n
  io.c0_ddr4_ck_t        := top.io.c0_ddr4_ck_t
  io.c0_ddr4_ck_c        := top.io.c0_ddr4_ck_c
  io.c0_ddr4_reset_n     := top.io.c0_ddr4_reset_n
  // DDR4 inout ports are managed by netlist macro (xepic_ddr4_dc1)
  // Do not connect them to avoid VCOM terminal register requirement
  // attach(io.c0_ddr4_dm_dbi_n, top.io.c0_ddr4_dm_dbi_n)
  // attach(io.c0_ddr4_dq, top.io.c0_ddr4_dq)
  // attach(io.c0_ddr4_dqs_c, top.io.c0_ddr4_dqs_c)
  // attach(io.c0_ddr4_dqs_t, top.io.c0_ddr4_dqs_t)
  io.c0_ddr4_ui_clk      := top.io.c0_ddr4_ui_clk
  io.init_calib_complete := top.io.init_calib_complete
  io.ddr4_en_vtt         := top.io.ddr4_en_vtt
  io.ddr4_en_vddq        := top.io.ddr4_en_vddq
  io.ddr4_en_vcc2v5      := top.io.ddr4_en_vcc2v5
  top.io.power_good      := io.power_good
}

object ElaborateP2ETop extends App {
  import _root_.circt.stage.ChiselStage

  ChiselStage.emitSystemVerilogFile(
    new P2ETop,
    firtoolOpts = args,
    args = Array.empty
  )
}
