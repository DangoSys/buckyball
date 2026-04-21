package sims.pegasus

import chisel3._
import chisel3.util.HasBlackBoxInline

import pegasus.PegasusShell

class PegasusTop extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val pcie_refclk_p   = Input(Bool())
    val pcie_refclk_n   = Input(Bool())
    val pcie_sys_rst_n  = Input(Bool())
    val pcie_exp_txp    = Output(UInt(16.W))
    val pcie_exp_txn    = Output(UInt(16.W))
    val pcie_exp_rxp    = Input(UInt(16.W))
    val pcie_exp_rxn    = Input(UInt(16.W))
    val c0_sys_clk_p    = Input(Bool())
    val c0_sys_clk_n    = Input(Bool())
    val c0_ddr4_act_n   = Output(Bool())
    val c0_ddr4_adr     = Output(UInt(17.W))
    val c0_ddr4_ba      = Output(UInt(2.W))
    val c0_ddr4_bg      = Output(UInt(2.W))
    val c0_ddr4_cke     = Output(UInt(1.W))
    val c0_ddr4_odt     = Output(UInt(1.W))
    val c0_ddr4_cs_n    = Output(UInt(1.W))
    val c0_ddr4_ck_t    = Output(UInt(1.W))
    val c0_ddr4_ck_c    = Output(UInt(1.W))
    val c0_ddr4_reset_n = Output(Bool())
    val c0_ddr4_parity  = Output(Bool())
    val uart_txd        = Output(Bool())
  })

  setInline(
    "PegasusTop.v",
    """module PegasusTop(
      |  input         pcie_refclk_p,
      |  input         pcie_refclk_n,
      |  input         pcie_sys_rst_n,
      |  output [15:0] pcie_exp_txp,
      |  output [15:0] pcie_exp_txn,
      |  input  [15:0] pcie_exp_rxp,
      |  input  [15:0] pcie_exp_rxn,
      |  input         c0_sys_clk_p,
      |  input         c0_sys_clk_n,
      |  output        c0_ddr4_act_n,
      |  output [16:0] c0_ddr4_adr,
      |  output [1:0]  c0_ddr4_ba,
      |  output [1:0]  c0_ddr4_bg,
      |  output [0:0]  c0_ddr4_cke,
      |  output [0:0]  c0_ddr4_odt,
      |  output [0:0]  c0_ddr4_cs_n,
      |  output [0:0]  c0_ddr4_ck_t,
      |  output [0:0]  c0_ddr4_ck_c,
      |  output        c0_ddr4_reset_n,
      |  output        c0_ddr4_parity,
      |  inout  [71:0] c0_ddr4_dq,
      |  inout  [17:0] c0_ddr4_dqs_c,
      |  inout  [17:0] c0_ddr4_dqs_t,
      |  output        uart_txd
      |);
      |
      |  wire        soc_clk;
      |  wire        soc_reset;
      |  wire        cpu_hold_reset;
      |
      |  // mem_axi4: DigitalTop → shell crossbar → DDR4
      |  wire [3:0]  mem_awid;    wire [31:0] mem_awaddr;
      |  wire [7:0]  mem_awlen;   wire [2:0]  mem_awsize;
      |  wire [1:0]  mem_awburst; wire        mem_awvalid, mem_awready;
      |  wire [63:0] mem_wdata;   wire [7:0]  mem_wstrb;
      |  wire        mem_wlast,   mem_wvalid, mem_wready;
      |  wire [3:0]  mem_bid;     wire [1:0]  mem_bresp;
      |  wire        mem_bvalid,  mem_bready;
      |  wire [3:0]  mem_arid;    wire [31:0] mem_araddr;
      |  wire [7:0]  mem_arlen;   wire [2:0]  mem_arsize;
      |  wire [1:0]  mem_arburst; wire        mem_arvalid, mem_arready;
      |  wire [3:0]  mem_rid;     wire [63:0] mem_rdata;
      |  wire [1:0]  mem_rresp;   wire        mem_rlast, mem_rvalid, mem_rready;
      |
      |  // mmio_axi4: DigitalTop → shell SCU
      |  wire [3:0]  mmio_awid;    wire [31:0] mmio_awaddr;
      |  wire [7:0]  mmio_awlen;   wire [2:0]  mmio_awsize;
      |  wire [1:0]  mmio_awburst; wire        mmio_awvalid, mmio_awready;
      |  wire [63:0] mmio_wdata;   wire [7:0]  mmio_wstrb;
      |  wire        mmio_wlast,   mmio_wvalid, mmio_wready;
      |  wire [3:0]  mmio_bid;     wire [1:0]  mmio_bresp;
      |  wire        mmio_bvalid,  mmio_bready;
      |  wire [3:0]  mmio_arid;    wire [31:0] mmio_araddr;
      |  wire [7:0]  mmio_arlen;   wire [2:0]  mmio_arsize;
      |  wire [1:0]  mmio_arburst; wire        mmio_arvalid, mmio_arready;
      |  wire [3:0]  mmio_rid;     wire [63:0] mmio_rdata;
      |  wire [1:0]  mmio_rresp;   wire        mmio_rlast, mmio_rvalid, mmio_rready;
      |
      |  PegasusShell shell (
      |    .pcie_refclk_p   (pcie_refclk_p),
      |    .pcie_refclk_n   (pcie_refclk_n),
      |    .pcie_sys_rst_n  (pcie_sys_rst_n),
      |    .pcie_exp_txp    (pcie_exp_txp),
      |    .pcie_exp_txn    (pcie_exp_txn),
      |    .pcie_exp_rxp    (pcie_exp_rxp),
      |    .pcie_exp_rxn    (pcie_exp_rxn),
      |    .c0_sys_clk_p    (c0_sys_clk_p),
      |    .c0_sys_clk_n    (c0_sys_clk_n),
      |    .c0_ddr4_act_n   (c0_ddr4_act_n),
      |    .c0_ddr4_adr     (c0_ddr4_adr),
      |    .c0_ddr4_ba      (c0_ddr4_ba),
      |    .c0_ddr4_bg      (c0_ddr4_bg),
      |    .c0_ddr4_cke     (c0_ddr4_cke),
      |    .c0_ddr4_odt     (c0_ddr4_odt),
      |    .c0_ddr4_cs_n    (c0_ddr4_cs_n),
      |    .c0_ddr4_ck_t    (c0_ddr4_ck_t),
      |    .c0_ddr4_ck_c    (c0_ddr4_ck_c),
      |    .c0_ddr4_reset_n (c0_ddr4_reset_n),
      |    .c0_ddr4_parity  (c0_ddr4_parity),
      |    .c0_ddr4_dq      (c0_ddr4_dq),
      |    .c0_ddr4_dqs_c   (c0_ddr4_dqs_c),
      |    .c0_ddr4_dqs_t   (c0_ddr4_dqs_t),
      |    .dut_clk         (soc_clk),
      |    .dut_reset       (soc_reset),
      |    .cpu_hold_reset  (cpu_hold_reset),
      |    .uart_tx         (uart_txd),
      |    // mem_axi4 → crossbar → DDR4
      |    .chip_mem_awid   (mem_awid),    .chip_mem_awaddr (mem_awaddr),
      |    .chip_mem_awlen  (mem_awlen),   .chip_mem_awsize (mem_awsize),
      |    .chip_mem_awburst(mem_awburst), .chip_mem_awvalid(mem_awvalid),
      |    .chip_mem_awready(mem_awready),
      |    .chip_mem_wdata  (mem_wdata),   .chip_mem_wstrb  (mem_wstrb),
      |    .chip_mem_wlast  (mem_wlast),   .chip_mem_wvalid (mem_wvalid),
      |    .chip_mem_wready (mem_wready),
      |    .chip_mem_bid    (mem_bid),     .chip_mem_bresp  (mem_bresp),
      |    .chip_mem_bvalid (mem_bvalid),  .chip_mem_bready (mem_bready),
      |    .chip_mem_arid   (mem_arid),    .chip_mem_araddr (mem_araddr),
      |    .chip_mem_arlen  (mem_arlen),   .chip_mem_arsize (mem_arsize),
      |    .chip_mem_arburst(mem_arburst), .chip_mem_arvalid(mem_arvalid),
      |    .chip_mem_arready(mem_arready),
      |    .chip_mem_rid    (mem_rid),     .chip_mem_rdata  (mem_rdata),
      |    .chip_mem_rresp  (mem_rresp),   .chip_mem_rlast  (mem_rlast),
      |    .chip_mem_rvalid (mem_rvalid),  .chip_mem_rready (mem_rready),
      |    // mmio_axi4 → SCU
      |    .mmio_awid       (mmio_awid),    .mmio_awaddr   (mmio_awaddr),
      |    .mmio_awlen      (mmio_awlen),   .mmio_awsize   (mmio_awsize),
      |    .mmio_awburst    (mmio_awburst), .mmio_awvalid  (mmio_awvalid),
      |    .mmio_awready    (mmio_awready),
      |    .mmio_wdata      (mmio_wdata),   .mmio_wstrb    (mmio_wstrb),
      |    .mmio_wlast      (mmio_wlast),   .mmio_wvalid   (mmio_wvalid),
      |    .mmio_wready     (mmio_wready),
      |    .mmio_bid        (mmio_bid),     .mmio_bresp    (mmio_bresp),
      |    .mmio_bvalid     (mmio_bvalid),  .mmio_bready   (mmio_bready),
      |    .mmio_arid       (mmio_arid),    .mmio_araddr   (mmio_araddr),
      |    .mmio_arlen      (mmio_arlen),   .mmio_arsize   (mmio_arsize),
      |    .mmio_arburst    (mmio_arburst), .mmio_arvalid  (mmio_arvalid),
      |    .mmio_arready    (mmio_arready),
      |    .mmio_rid        (mmio_rid),     .mmio_rdata    (mmio_rdata),
      |    .mmio_rresp      (mmio_rresp),   .mmio_rlast   (mmio_rlast),
      |    .mmio_rvalid     (mmio_rvalid),  .mmio_rready   (mmio_rready)
      |  );
      |
      |  DigitalTop soc (
      |    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock (soc_clk),
      |    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_reset (soc_reset),
      |    .resetctrl_hartIsInReset_0 (cpu_hold_reset),
      |    .debug_clock               (soc_clk),
      |    .debug_reset               (soc_reset),
      |    .debug_systemjtag_reset    (soc_reset),
      |    .debug_systemjtag_jtag_TCK (1'b0),
      |    .debug_systemjtag_jtag_TMS (1'b1),
      |    .debug_systemjtag_jtag_TDI (1'b1),
      |    .debug_dmactiveAck         (1'b0),
      |    .uart_0_txd                (uart_txd),
      |    .uart_0_rxd                (1'b1),
      |    // mem_axi4 → shell → DDR4
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
      |    // mmio_axi4 → shell → SCU
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
      |    // No inbound DMA port (removed WithCustomSlavePort)
      |    .serial_tl_0_in_valid      (1'b0),
      |    .serial_tl_0_in_bits_phit  (32'h0),
      |    .serial_tl_0_out_ready     (1'b0),
      |    .serial_tl_0_clock_in      (1'b0),
      |    .custom_boot               (1'b0)
      |  );
      |
      |endmodule
      |""".stripMargin
  )
}

object ElaboratePegasusTop extends App {
  import _root_.circt.stage.ChiselStage

  class PegasusTopWrapper extends RawModule {

    val io = IO(new Bundle {
      val pcie_refclk_p   = Input(Bool())
      val pcie_refclk_n   = Input(Bool())
      val pcie_sys_rst_n  = Input(Bool())
      val pcie_exp_txp    = Output(UInt(16.W))
      val pcie_exp_txn    = Output(UInt(16.W))
      val pcie_exp_rxp    = Input(UInt(16.W))
      val pcie_exp_rxn    = Input(UInt(16.W))
      val c0_sys_clk_p    = Input(Bool())
      val c0_sys_clk_n    = Input(Bool())
      val c0_ddr4_act_n   = Output(Bool())
      val c0_ddr4_adr     = Output(UInt(17.W))
      val c0_ddr4_ba      = Output(UInt(2.W))
      val c0_ddr4_bg      = Output(UInt(2.W))
      val c0_ddr4_cke     = Output(UInt(1.W))
      val c0_ddr4_odt     = Output(UInt(1.W))
      val c0_ddr4_cs_n    = Output(UInt(1.W))
      val c0_ddr4_ck_t    = Output(UInt(1.W))
      val c0_ddr4_ck_c    = Output(UInt(1.W))
      val c0_ddr4_reset_n = Output(Bool())
      val c0_ddr4_parity  = Output(Bool())
      val uart_txd        = Output(Bool())
    })

    val top = Module(new PegasusTop)
    top.io.pcie_refclk_p  := io.pcie_refclk_p
    top.io.pcie_refclk_n  := io.pcie_refclk_n
    top.io.pcie_sys_rst_n := io.pcie_sys_rst_n
    top.io.pcie_exp_rxp   := io.pcie_exp_rxp
    top.io.pcie_exp_rxn   := io.pcie_exp_rxn
    top.io.c0_sys_clk_p   := io.c0_sys_clk_p
    top.io.c0_sys_clk_n   := io.c0_sys_clk_n
    io.pcie_exp_txp       := top.io.pcie_exp_txp
    io.pcie_exp_txn       := top.io.pcie_exp_txn
    io.c0_ddr4_act_n      := top.io.c0_ddr4_act_n
    io.c0_ddr4_adr        := top.io.c0_ddr4_adr
    io.c0_ddr4_ba         := top.io.c0_ddr4_ba
    io.c0_ddr4_bg         := top.io.c0_ddr4_bg
    io.c0_ddr4_cke        := top.io.c0_ddr4_cke
    io.c0_ddr4_odt        := top.io.c0_ddr4_odt
    io.c0_ddr4_cs_n       := top.io.c0_ddr4_cs_n
    io.c0_ddr4_ck_t       := top.io.c0_ddr4_ck_t
    io.c0_ddr4_ck_c       := top.io.c0_ddr4_ck_c
    io.c0_ddr4_reset_n    := top.io.c0_ddr4_reset_n
    io.c0_ddr4_parity     := top.io.c0_ddr4_parity
    io.uart_txd           := top.io.uart_txd
  }

  ChiselStage.emitSystemVerilogFile(
    new PegasusTopWrapper,
    firtoolOpts = args,
    args = Array.empty
  )
}
