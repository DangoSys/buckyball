package sims.pegasus

import chisel3._

import org.chipsalliance.cde.config.Parameters

import chipyard.harness.HasHarnessInstantiators
import chipyard.iobinders.AXI4MemPort

import pegasus.PegasusShell

class PegasusHarness(implicit val p: Parameters) extends Module with HasHarnessInstantiators {

  val io = IO(new Bundle {
    val pcie_refclk_p  = Input(Bool())
    val pcie_refclk_n  = Input(Bool())
    val pcie_sys_rst_n = Input(Bool())
    val pcie_exp_txp   = Output(UInt(16.W))
    val pcie_exp_txn   = Output(UInt(16.W))
    val pcie_exp_rxp   = Input(UInt(16.W))
    val pcie_exp_rxn   = Input(UInt(16.W))
  })

  val pegasusShell = Module(new PegasusShell)
  pegasusShell.io.pcie_refclk_p  := io.pcie_refclk_p
  pegasusShell.io.pcie_refclk_n  := io.pcie_refclk_n
  pegasusShell.io.pcie_sys_rst_n := io.pcie_sys_rst_n
  io.pcie_exp_txp                := pegasusShell.io.pcie_exp_txp
  io.pcie_exp_txn                := pegasusShell.io.pcie_exp_txn
  pegasusShell.io.pcie_exp_rxp   := io.pcie_exp_rxp
  pegasusShell.io.pcie_exp_rxn   := io.pcie_exp_rxn
  pegasusShell.io.c0_sys_clk_p   := false.B
  pegasusShell.io.c0_sys_clk_n   := false.B

  pegasusShell.io.uart_tx := true.B

  // Default tie-offs for chip_mem (overridden by connectChipMem)
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

  // Default tie-offs for mmio (overridden by connectChipMMIO)
  pegasusShell.io.mmio_awid    := 0.U
  pegasusShell.io.mmio_awaddr  := 0.U
  pegasusShell.io.mmio_awlen   := 0.U
  pegasusShell.io.mmio_awsize  := 0.U
  pegasusShell.io.mmio_awburst := 0.U
  pegasusShell.io.mmio_awvalid := false.B
  pegasusShell.io.mmio_wdata   := 0.U
  pegasusShell.io.mmio_wstrb   := 0.U
  pegasusShell.io.mmio_wlast   := false.B
  pegasusShell.io.mmio_wvalid  := false.B
  pegasusShell.io.mmio_bready  := false.B
  pegasusShell.io.mmio_arid    := 0.U
  pegasusShell.io.mmio_araddr  := 0.U
  pegasusShell.io.mmio_arlen   := 0.U
  pegasusShell.io.mmio_arsize  := 0.U
  pegasusShell.io.mmio_arburst := 0.U
  pegasusShell.io.mmio_arvalid := false.B
  pegasusShell.io.mmio_rready  := false.B

  def referenceClockFreqMHz: Double = 150.0
  def referenceClock:        Clock  = pegasusShell.io.dut_clk
  def referenceReset:        Reset  = pegasusShell.io.dut_reset.asAsyncReset

  val success = WireInit(false.B)

  val lazyDuts = instantiateChipTops()

  def connectChipMem(port: AXI4MemPort): Unit = {
    val axi = port.io.bits
    pegasusShell.io.chip_mem_awid    := axi.aw.bits.id.asTypeOf(UInt(4.W))
    pegasusShell.io.chip_mem_awaddr  := axi.aw.bits.addr.asTypeOf(UInt(32.W))
    pegasusShell.io.chip_mem_awlen   := axi.aw.bits.len
    pegasusShell.io.chip_mem_awsize  := axi.aw.bits.size
    pegasusShell.io.chip_mem_awburst := axi.aw.bits.burst
    pegasusShell.io.chip_mem_awvalid := axi.aw.valid
    axi.aw.ready                     := pegasusShell.io.chip_mem_awready

    pegasusShell.io.chip_mem_wdata  := axi.w.bits.data.asTypeOf(UInt(64.W))
    pegasusShell.io.chip_mem_wstrb  := axi.w.bits.strb.asTypeOf(UInt(8.W))
    pegasusShell.io.chip_mem_wlast  := axi.w.bits.last
    pegasusShell.io.chip_mem_wvalid := axi.w.valid
    axi.w.ready                     := pegasusShell.io.chip_mem_wready

    axi.b.bits.id                   := pegasusShell.io.chip_mem_bid.asTypeOf(axi.b.bits.id)
    axi.b.bits.resp                 := pegasusShell.io.chip_mem_bresp
    axi.b.valid                     := pegasusShell.io.chip_mem_bvalid
    pegasusShell.io.chip_mem_bready := axi.b.ready

    pegasusShell.io.chip_mem_arid    := axi.ar.bits.id.asTypeOf(UInt(4.W))
    pegasusShell.io.chip_mem_araddr  := axi.ar.bits.addr.asTypeOf(UInt(32.W))
    pegasusShell.io.chip_mem_arlen   := axi.ar.bits.len
    pegasusShell.io.chip_mem_arsize  := axi.ar.bits.size
    pegasusShell.io.chip_mem_arburst := axi.ar.bits.burst
    pegasusShell.io.chip_mem_arvalid := axi.ar.valid
    axi.ar.ready                     := pegasusShell.io.chip_mem_arready

    axi.r.bits.id                   := pegasusShell.io.chip_mem_rid.asTypeOf(axi.r.bits.id)
    axi.r.bits.data                 := pegasusShell.io.chip_mem_rdata.asTypeOf(axi.r.bits.data)
    axi.r.bits.resp                 := pegasusShell.io.chip_mem_rresp
    axi.r.bits.last                 := pegasusShell.io.chip_mem_rlast
    axi.r.valid                     := pegasusShell.io.chip_mem_rvalid
    pegasusShell.io.chip_mem_rready := axi.r.ready
  }

}
