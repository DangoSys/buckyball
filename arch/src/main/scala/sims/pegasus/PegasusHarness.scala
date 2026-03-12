package sims.pegasus

import chisel3._
import chisel3.util._

import org.chipsalliance.cde.config.{Parameters}
import freechips.rocketchip.util.{ResetCatchAndSync}

import chipyard.harness.{HasHarnessInstantiators}
import chipyard.iobinders.{AXI4MemPort}

import pegasus._

// PegasusHarness: top-level Chisel Module for AU280 FPGA.
// Integrates PegasusShell (FPGA IPs) with ChipTop (SoC DUT) via Chipyard
// harness infrastructure.
//
// PCIe and HBM2 reference clock pins are the top-level IOs.
// ChipTop is instantiated inside via instantiateChipTops().
// HarnessBinders connect ChipTop's exposed ports (AXI4MemPort, UARTPort, etc.)
// to pegasusShell's interface signals.
//
class PegasusHarness(implicit val p: Parameters) extends Module with HasHarnessInstantiators {

  val io = IO(new Bundle {
    // PCIe physical interface
    val pcie_sys_clk    = Input(Clock())
    val pcie_sys_clk_gt = Input(Clock())
    val pcie_sys_rst_n  = Input(Bool())
    val pcie_exp_txp    = Output(UInt(16.W))
    val pcie_exp_txn    = Output(UInt(16.W))
    val pcie_exp_rxp    = Input(UInt(16.W))
    val pcie_exp_rxn    = Input(UInt(16.W))

    // HBM2 reference clock (100 MHz)
    val hbm_ref_clk = Input(Clock())
  })

  // --- Instantiate PegasusShell ---
  val pegasusShell = Module(new PegasusShell)
  pegasusShell.io.pcie_sys_clk    := io.pcie_sys_clk
  pegasusShell.io.pcie_sys_clk_gt := io.pcie_sys_clk_gt
  pegasusShell.io.pcie_sys_rst_n  := io.pcie_sys_rst_n
  io.pcie_exp_txp                 := pegasusShell.io.pcie_exp_txp
  io.pcie_exp_txn                 := pegasusShell.io.pcie_exp_txn
  pegasusShell.io.pcie_exp_rxp    := io.pcie_exp_rxp
  pegasusShell.io.pcie_exp_rxn    := io.pcie_exp_rxn
  pegasusShell.io.hbm_ref_clk     := io.hbm_ref_clk

  // UART TX: default idle high until ChipTop provides real signal
  pegasusShell.io.uart_tx := true.B

  // Chip mem: tie off until ChipTop provides real signals
  pegasusShell.io.chip_mem_awid    := 0.U
  pegasusShell.io.chip_mem_awaddr  := 0.U
  pegasusShell.io.chip_mem_awlen   := 0.U
  pegasusShell.io.chip_mem_awsize  := 0.U
  pegasusShell.io.chip_mem_awburst := 0.U
  pegasusShell.io.chip_mem_awvalid := false.B
  pegasusShell.io.chip_mem_wdata   := 0.U
  pegasusShell.io.chip_mem_wstrb   := 0.U
  pegasusShell.io.chip_mem_wlast   := false.B
  pegasusShell.io.chip_mem_wvalid  := false.B
  pegasusShell.io.chip_mem_bready  := false.B
  pegasusShell.io.chip_mem_arid    := 0.U
  pegasusShell.io.chip_mem_araddr  := 0.U
  pegasusShell.io.chip_mem_arlen   := 0.U
  pegasusShell.io.chip_mem_arsize  := 0.U
  pegasusShell.io.chip_mem_arburst := 0.U
  pegasusShell.io.chip_mem_arvalid := false.B
  pegasusShell.io.chip_mem_rready  := false.B

  // --- HasHarnessInstantiators required interface ---
  def referenceClockFreqMHz: Double = 200.0
  def referenceClock:        Clock  = pegasusShell.io.dut_clk
  def referenceReset:        Reset  = pegasusShell.io.dut_reset.asAsyncReset

  val success = WireInit(false.B)

  // --- Instantiate ChipTop and apply HarnessBinders ---
  // HarnessBinders for UART and AXI4Mem will override the defaults above
  // by calling connectChipMem() and driving pegasusShell.io.uart_tx
  val lazyDuts = instantiateChipTops()

  // Called by WithPegasusAXIMem HarnessBinder to connect ChipTop ExtMem AXI4
  def connectChipMem(port: AXI4MemPort): Unit = {
    val axi = port.io.bits
    // Write address channel
    pegasusShell.io.chip_mem_awid    := axi.aw.bits.id.asTypeOf(UInt(6.W))
    pegasusShell.io.chip_mem_awaddr  := axi.aw.bits.addr.asTypeOf(UInt(33.W))
    pegasusShell.io.chip_mem_awlen   := axi.aw.bits.len
    pegasusShell.io.chip_mem_awsize  := axi.aw.bits.size
    pegasusShell.io.chip_mem_awburst := axi.aw.bits.burst
    pegasusShell.io.chip_mem_awvalid := axi.aw.valid
    axi.aw.ready                     := pegasusShell.io.chip_mem_awready
    // Write data channel
    pegasusShell.io.chip_mem_wdata   := axi.w.bits.data.asTypeOf(UInt(256.W))
    pegasusShell.io.chip_mem_wstrb   := axi.w.bits.strb.asTypeOf(UInt(32.W))
    pegasusShell.io.chip_mem_wlast   := axi.w.bits.last
    pegasusShell.io.chip_mem_wvalid  := axi.w.valid
    axi.w.ready                      := pegasusShell.io.chip_mem_wready
    // Write response channel
    axi.b.bits.id                    := pegasusShell.io.chip_mem_bid.asTypeOf(axi.b.bits.id)
    axi.b.bits.resp                  := pegasusShell.io.chip_mem_bresp
    axi.b.valid                      := pegasusShell.io.chip_mem_bvalid
    pegasusShell.io.chip_mem_bready  := axi.b.ready
    // Read address channel
    pegasusShell.io.chip_mem_arid    := axi.ar.bits.id.asTypeOf(UInt(6.W))
    pegasusShell.io.chip_mem_araddr  := axi.ar.bits.addr.asTypeOf(UInt(33.W))
    pegasusShell.io.chip_mem_arlen   := axi.ar.bits.len
    pegasusShell.io.chip_mem_arsize  := axi.ar.bits.size
    pegasusShell.io.chip_mem_arburst := axi.ar.bits.burst
    pegasusShell.io.chip_mem_arvalid := axi.ar.valid
    axi.ar.ready                     := pegasusShell.io.chip_mem_arready
    // Read data channel
    axi.r.bits.id                    := pegasusShell.io.chip_mem_rid.asTypeOf(axi.r.bits.id)
    axi.r.bits.data                  := pegasusShell.io.chip_mem_rdata.asTypeOf(axi.r.bits.data)
    axi.r.bits.resp                  := pegasusShell.io.chip_mem_rresp
    axi.r.bits.last                  := pegasusShell.io.chip_mem_rlast
    axi.r.valid                      := pegasusShell.io.chip_mem_rvalid
    pegasusShell.io.chip_mem_rready  := axi.r.ready
  }

}
