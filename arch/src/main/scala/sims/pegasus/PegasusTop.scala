package sims.pegasus

import chisel3._
import chisel3.util.HasBlackBoxInline

import pegasus.PegasusShell

// PegasusTop — Vivado bitstream top-level for AU280 bring-up.
// Inline Verilog BlackBox that connects PegasusShell (XDMA) to DigitalTop (SoC).
// Skips ChipTop/PegasusHarness harness layers entirely — no simulation primitives.
class PegasusTop extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val pcie_sys_clk    = Input(Clock())
    val pcie_sys_clk_gt = Input(Clock())
    val pcie_sys_rst_n  = Input(Bool())
    val pcie_exp_txp    = Output(UInt(16.W))
    val pcie_exp_txn    = Output(UInt(16.W))
    val pcie_exp_rxp    = Input(UInt(16.W))
    val pcie_exp_rxn    = Input(UInt(16.W))
    val hbm_ref_clk     = Input(Clock())
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
      |  input         hbm_ref_clk,
      |  output        uart_txd
      |);
      |
      |  wire soc_clk;
      |  wire soc_reset;
      |
      |  PegasusShell shell (
      |    .pcie_sys_clk    (pcie_sys_clk),
      |    .pcie_sys_clk_gt (pcie_sys_clk_gt),
      |    .pcie_sys_rst_n  (pcie_sys_rst_n),
      |    .pcie_exp_txp    (pcie_exp_txp),
      |    .pcie_exp_txn    (pcie_exp_txn),
      |    .pcie_exp_rxp    (pcie_exp_rxp),
      |    .pcie_exp_rxn    (pcie_exp_rxn),
      |    .hbm_ref_clk     (hbm_ref_clk),
      |    .dut_clk         (soc_clk),
      |    .dut_reset       (soc_reset),
      |    .uart_tx         (1'b1),
      |    .chip_mem_awid(6'h0), .chip_mem_awaddr(33'h0), .chip_mem_awlen(8'h0),
      |    .chip_mem_awsize(3'h0), .chip_mem_awburst(2'h0), .chip_mem_awvalid(1'b0),
      |    .chip_mem_wdata(256'h0), .chip_mem_wstrb(32'h0),
      |    .chip_mem_wlast(1'b0), .chip_mem_wvalid(1'b0), .chip_mem_bready(1'b0),
      |    .chip_mem_arid(6'h0), .chip_mem_araddr(33'h0), .chip_mem_arlen(8'h0),
      |    .chip_mem_arsize(3'h0), .chip_mem_arburst(2'h0), .chip_mem_arvalid(1'b0),
      |    .chip_mem_rready(1'b0)
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
      |    .mem_axi4_0_aw_ready       (1'b0),
      |    .mem_axi4_0_w_ready        (1'b0),
      |    .mem_axi4_0_b_valid        (1'b0),
      |    .mem_axi4_0_b_bits_id      (4'h0),
      |    .mem_axi4_0_b_bits_resp    (2'h0),
      |    .mem_axi4_0_ar_ready       (1'b0),
      |    .mem_axi4_0_r_valid        (1'b0),
      |    .mem_axi4_0_r_bits_id      (4'h0),
      |    .mem_axi4_0_r_bits_data    (64'h0),
      |    .mem_axi4_0_r_bits_resp    (2'h0),
      |    .mem_axi4_0_r_bits_last    (1'b0),
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

  // PegasusTop is a BlackBox with inline Verilog — wrap in a thin RawModule to elaborate
  class PegasusTopWrapper extends RawModule {

    val io = IO(new Bundle {
      val pcie_sys_clk    = Input(Clock())
      val pcie_sys_clk_gt = Input(Clock())
      val pcie_sys_rst_n  = Input(Bool())
      val pcie_exp_txp    = Output(UInt(16.W))
      val pcie_exp_txn    = Output(UInt(16.W))
      val pcie_exp_rxp    = Input(UInt(16.W))
      val pcie_exp_rxn    = Input(UInt(16.W))
      val hbm_ref_clk     = Input(Clock())
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
    top.io.hbm_ref_clk     := io.hbm_ref_clk
    io.uart_txd            := top.io.uart_txd
  }

  ChiselStage.emitSystemVerilogFile(
    new PegasusTopWrapper,
    firtoolOpts = args,
    args = Array.empty
  )
}
