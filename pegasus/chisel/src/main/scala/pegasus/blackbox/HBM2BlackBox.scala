package pegasus.blackbox

import chisel3._
import chisel3.util._

// AXI4 HBM2 slave port bundle (one pseudo-channel)
// Data width: 256-bit, addr width: 33-bit, id width: 6-bit
class HBM2AXI4Bundle extends Bundle {
  // Write address channel
  val awid    = Input(UInt(6.W))
  val awaddr  = Input(UInt(33.W))
  val awlen   = Input(UInt(4.W))   // HBM2 supports burst length up to 16
  val awsize  = Input(UInt(3.W))
  val awburst = Input(UInt(2.W))
  val awvalid = Input(Bool())
  val awready = Output(Bool())

  // Write data channel
  val wdata   = Input(UInt(256.W))
  val wstrb   = Input(UInt(32.W))
  val wlast   = Input(Bool())
  val wvalid  = Input(Bool())
  val wready  = Output(Bool())

  // Write response channel
  val bid     = Output(UInt(6.W))
  val bresp   = Output(UInt(2.W))
  val bvalid  = Output(Bool())
  val bready  = Input(Bool())

  // Read address channel
  val arid    = Input(UInt(6.W))
  val araddr  = Input(UInt(33.W))
  val arlen   = Input(UInt(4.W))
  val arsize  = Input(UInt(3.W))
  val arburst = Input(UInt(2.W))
  val arvalid = Input(Bool())
  val arready = Output(Bool())

  // Read data channel
  val rid     = Output(UInt(6.W))
  val rdata   = Output(UInt(256.W))
  val rresp   = Output(UInt(2.W))
  val rlast   = Output(Bool())
  val rvalid  = Output(Bool())
  val rready  = Input(Bool())
}

// Xilinx HBM2 IP black box for AU280 (xcu280)
// AU280 has 2 HBM stacks, each with 8 pseudo-channels.
// MVP uses Stack 0, PC0 only.
// AXI interface: 256-bit data, 33-bit address, 6-bit ID, 250 MHz
class HBM2BlackBox extends BlackBox {
  override def desiredName = "hbm_0"
  val io = IO(new Bundle {
    // Reference clocks for HBM2 (provided by MMCM)
    val HBM_REF_CLK_0   = Input(Clock())   // 100 MHz reference for stack 0

    // AXI interface clock (250 MHz, from XDMA axi_aclk)
    val AXI_00_ACLK     = Input(Clock())
    val AXI_00_ARESET_N = Input(Bool())

    // AXI slave port for PC0 (Stack 0, Pseudo-channel 0)
    val AXI_00_AWID     = Input(UInt(6.W))
    val AXI_00_AWADDR   = Input(UInt(33.W))
    val AXI_00_AWLEN    = Input(UInt(4.W))
    val AXI_00_AWSIZE   = Input(UInt(3.W))
    val AXI_00_AWBURST  = Input(UInt(2.W))
    val AXI_00_AWVALID  = Input(Bool())
    val AXI_00_AWREADY  = Output(Bool())

    val AXI_00_WDATA    = Input(UInt(256.W))
    val AXI_00_WSTRB    = Input(UInt(32.W))
    val AXI_00_WLAST    = Input(Bool())
    val AXI_00_WVALID   = Input(Bool())
    val AXI_00_WREADY   = Output(Bool())

    val AXI_00_BID      = Output(UInt(6.W))
    val AXI_00_BRESP    = Output(UInt(2.W))
    val AXI_00_BVALID   = Output(Bool())
    val AXI_00_BREADY   = Input(Bool())

    val AXI_00_ARID     = Input(UInt(6.W))
    val AXI_00_ARADDR   = Input(UInt(33.W))
    val AXI_00_ARLEN    = Input(UInt(4.W))
    val AXI_00_ARSIZE   = Input(UInt(3.W))
    val AXI_00_ARBURST  = Input(UInt(2.W))
    val AXI_00_ARVALID  = Input(Bool())
    val AXI_00_ARREADY  = Output(Bool())

    val AXI_00_RID      = Output(UInt(6.W))
    val AXI_00_RDATA    = Output(UInt(256.W))
    val AXI_00_RRESP    = Output(UInt(2.W))
    val AXI_00_RLAST    = Output(Bool())
    val AXI_00_RVALID   = Output(Bool())
    val AXI_00_RREADY   = Input(Bool())

    // APB (configuration) interface
    val APB_0_PWDATA    = Input(UInt(32.W))
    val APB_0_PADDR     = Input(UInt(22.W))
    val APB_0_PCLK      = Input(Clock())
    val APB_0_PENABLE   = Input(Bool())
    val APB_0_PRESET_N  = Input(Bool())
    val APB_0_PSEL      = Input(Bool())
    val APB_0_PWRITE    = Input(Bool())
    val APB_0_PRDATA    = Output(UInt(32.W))
    val APB_0_PREADY    = Output(Bool())
    val APB_0_PSLVERR   = Output(Bool())

    // HBM2 initialization status
    val apb_complete_0  = Output(Bool())
    val DRAM_0_STAT_CATTRIP = Output(Bool())
    val DRAM_0_STAT_TEMP    = Output(UInt(7.W))
  })
}
