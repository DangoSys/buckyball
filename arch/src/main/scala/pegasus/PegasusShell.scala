package pegasus

import chisel3._

class PegasusShell extends Module {
  val io = IO(new Bundle {
    val pcie_sys_clk    = Input(Clock())
    val pcie_sys_clk_gt = Input(Clock())
    val pcie_sys_rst_n  = Input(Bool())
    val pcie_exp_txp    = Output(UInt(16.W))
    val pcie_exp_txn    = Output(UInt(16.W))
    val pcie_exp_rxp    = Input(UInt(16.W))
    val pcie_exp_rxn    = Input(UInt(16.W))
    val hbm_ref_clk     = Input(Clock())

    val uart_tx = Input(Bool())

    val chip_mem_awid    = Input(UInt(6.W))
    val chip_mem_awaddr  = Input(UInt(33.W))
    val chip_mem_awlen   = Input(UInt(8.W))
    val chip_mem_awsize  = Input(UInt(3.W))
    val chip_mem_awburst = Input(UInt(2.W))
    val chip_mem_awvalid = Input(Bool())
    val chip_mem_awready = Output(Bool())

    val chip_mem_wdata  = Input(UInt(256.W))
    val chip_mem_wstrb  = Input(UInt(32.W))
    val chip_mem_wlast  = Input(Bool())
    val chip_mem_wvalid = Input(Bool())
    val chip_mem_wready = Output(Bool())

    val chip_mem_bid    = Output(UInt(6.W))
    val chip_mem_bresp  = Output(UInt(2.W))
    val chip_mem_bvalid = Output(Bool())
    val chip_mem_bready = Input(Bool())

    val chip_mem_arid    = Input(UInt(6.W))
    val chip_mem_araddr  = Input(UInt(33.W))
    val chip_mem_arlen   = Input(UInt(8.W))
    val chip_mem_arsize  = Input(UInt(3.W))
    val chip_mem_arburst = Input(UInt(2.W))
    val chip_mem_arvalid = Input(Bool())
    val chip_mem_arready = Output(Bool())

    val chip_mem_rid    = Output(UInt(6.W))
    val chip_mem_rdata  = Output(UInt(256.W))
    val chip_mem_rresp  = Output(UInt(2.W))
    val chip_mem_rlast  = Output(Bool())
    val chip_mem_rvalid = Output(Bool())
    val chip_mem_rready = Input(Bool())

    val dut_clk   = Output(Clock())
    val dut_reset = Output(Bool())
  })

  io.pcie_exp_txp := 0.U
  io.pcie_exp_txn := 0.U

  io.chip_mem_awready := false.B
  io.chip_mem_wready  := false.B
  io.chip_mem_bid     := 0.U
  io.chip_mem_bresp   := 0.U
  io.chip_mem_bvalid  := false.B
  io.chip_mem_arready := false.B
  io.chip_mem_rid     := 0.U
  io.chip_mem_rdata   := 0.U
  io.chip_mem_rresp   := 0.U
  io.chip_mem_rlast   := false.B
  io.chip_mem_rvalid  := false.B

  io.dut_clk   := io.pcie_sys_clk
  io.dut_reset := !io.pcie_sys_rst_n
}
