package sims.pegasus

import chisel3._
import chisel3.util.HasBlackBoxInline

import pegasus.PegasusShell

// PegasusTop — Vivado bitstream top-level for AU280.
// XDMA (DMA mode) + DDR4 + SCU + DigitalTop SoC.
// H2C pwrite offset = SoC paddr - 0x80000000 (DDR4 base).
class PegasusTop extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val pcie_sys_clk    = Input(Clock())
    val pcie_sys_clk_gt = Input(Clock())
    val pcie_sys_rst_n  = Input(Bool())
    val pcie_exp_txp    = Output(UInt(16.W))
    val pcie_exp_txn    = Output(UInt(16.W))
    val pcie_exp_rxp    = Input(UInt(16.W))
    val pcie_exp_rxn    = Input(UInt(16.W))
    // DDR4 physical pins are auto-handled by Vivado board interface — no ports here.
    val uart_txd        = Output(Bool())
  })

  setInline(
    "PegasusTop.v",
    """module PegasusTop(
      |  input         pcie_sys_clk,
      |  input         pcie_sys_clk_gt,
      |  input         pcie_sys_rst_n,
      |  output [15:0] pcie_exp_txp,
      |  output [15:0] pcie_exp_txn,
      |  input  [15:0] pcie_exp_rxp,
      |  input  [15:0] pcie_exp_rxn,
      |  // DDR4 physical pins handled by Vivado board interface — no RTL ports here.
      |  output        uart_txd
      |);
      |
      |  wire        soc_clk;
      |  wire        soc_reset;
      |
      |  // mem_axi4_0 wires (64-bit, 32-bit addr, 4-bit id)
      |  wire [3:0]  mem_awid;
      |  wire [31:0] mem_awaddr;
      |  wire [7:0]  mem_awlen;
      |  wire [2:0]  mem_awsize;
      |  wire [1:0]  mem_awburst;
      |  wire        mem_awvalid, mem_awready;
      |  wire [63:0] mem_wdata;
      |  wire [7:0]  mem_wstrb;
      |  wire        mem_wlast, mem_wvalid, mem_wready;
      |  wire [3:0]  mem_bid;
      |  wire [1:0]  mem_bresp;
      |  wire        mem_bvalid, mem_bready;
      |  wire [3:0]  mem_arid;
      |  wire [31:0] mem_araddr;
      |  wire [7:0]  mem_arlen;
      |  wire [2:0]  mem_arsize;
      |  wire [1:0]  mem_arburst;
      |  wire        mem_arvalid, mem_arready;
      |  wire [3:0]  mem_rid;
      |  wire [63:0] mem_rdata;
      |  wire [1:0]  mem_rresp;
      |  wire        mem_rlast, mem_rvalid, mem_rready;
      |
      |  PegasusShell shell (
      |    .pcie_sys_clk    (pcie_sys_clk),
      |    .pcie_sys_clk_gt (pcie_sys_clk_gt),
      |    .pcie_sys_rst_n  (pcie_sys_rst_n),
      |    .pcie_exp_txp    (pcie_exp_txp),
      |    .pcie_exp_txn    (pcie_exp_txn),
      |    .pcie_exp_rxp    (pcie_exp_rxp),
      |    .pcie_exp_rxn    (pcie_exp_rxn),
      |    // DDR4 physical pins: no connections — handled by Vivado board interface.
      |    .dut_clk         (soc_clk),
      |    .dut_reset       (soc_reset),
      |    .uart_tx         (1'b1),
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
      |    .chip_mem_rvalid (mem_rvalid),  .chip_mem_rready (mem_rready)
      |  );
      |
      |  DigitalTop soc (
      |    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_clock (soc_clk),
      |    .auto_chipyard_prcictrl_domain_reset_setter_clock_in_member_allClocks_uncore_reset (soc_reset),
      |    .resetctrl_hartIsInReset_0 (soc_reset),
      |    .debug_clock               (soc_clk),
      |    .debug_reset               (soc_reset),
      |    .debug_systemjtag_reset    (soc_reset),
      |    .debug_systemjtag_jtag_TCK (1'b0),
      |    .debug_systemjtag_jtag_TMS (1'b1),
      |    .debug_systemjtag_jtag_TDI (1'b1),
      |    .debug_dmactiveAck         (1'b0),
      |    .uart_0_txd                (uart_txd),
      |    .uart_0_rxd                (1'b1),
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
      |    .mmio_axi4_0_aw_ready      (1'b0),
      |    .mmio_axi4_0_w_ready       (1'b0),
      |    .mmio_axi4_0_b_valid       (1'b0),
      |    .mmio_axi4_0_b_bits_id     (4'h0),
      |    .mmio_axi4_0_b_bits_resp   (2'h0),
      |    .mmio_axi4_0_ar_ready      (1'b0),
      |    .mmio_axi4_0_r_valid       (1'b0),
      |    .mmio_axi4_0_r_bits_id     (4'h0),
      |    .mmio_axi4_0_r_bits_data   (64'h0),
      |    .mmio_axi4_0_r_bits_resp   (2'h0),
      |    .mmio_axi4_0_r_bits_last   (1'b0),
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
      val pcie_sys_clk    = Input(Clock())
      val pcie_sys_clk_gt = Input(Clock())
      val pcie_sys_rst_n  = Input(Bool())
      val pcie_exp_txp    = Output(UInt(16.W))
      val pcie_exp_txn    = Output(UInt(16.W))
      val pcie_exp_rxp    = Input(UInt(16.W))
      val pcie_exp_rxn    = Input(UInt(16.W))
      // DDR4 physical pins are auto-handled by Vivado board interface — no ports here.
      val uart_txd        = Output(Bool())
    })

    val top = Module(new PegasusTop)
    top.io.pcie_sys_clk    := io.pcie_sys_clk
    top.io.pcie_sys_clk_gt := io.pcie_sys_clk_gt
    top.io.pcie_sys_rst_n  := io.pcie_sys_rst_n
    top.io.pcie_exp_rxp    := io.pcie_exp_rxp
    top.io.pcie_exp_rxn    := io.pcie_exp_rxn
    io.pcie_exp_txp        := top.io.pcie_exp_txp
    io.pcie_exp_txn        := top.io.pcie_exp_txn
    io.uart_txd            := top.io.uart_txd
  }

  ChiselStage.emitSystemVerilogFile(
    new PegasusTopWrapper,
    firtoolOpts = args,
    args = Array.empty
  )
}
