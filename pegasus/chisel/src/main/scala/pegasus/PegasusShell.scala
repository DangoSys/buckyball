package pegasus

import chisel3._
import chisel3.util._
import pegasus.blackbox.{XDMABlackBox, HBM2BlackBox}

// AXI4 Crossbar black box (Vivado axi_crossbar IP)
// 2 masters (XDMA DMA + ChipTop ExtMem), 1 slave (HBM2 PC0)
// Data width: 256-bit (narrowed from XDMA's 512-bit via width converter)
// Address width: 33-bit (HBM2 address space)
class AXICrossbarBlackBox extends BlackBox {
  override def desiredName = "axi_crossbar_0"
  val io = IO(new Bundle {
    val aclk    = Input(Clock())
    val aresetn = Input(Bool())

    // Master port 0: XDMA DMA (width-converted to 256-bit)
    val s_axi_awid_0    = Input(UInt(6.W))
    val s_axi_awaddr_0  = Input(UInt(33.W))
    val s_axi_awlen_0   = Input(UInt(8.W))
    val s_axi_awsize_0  = Input(UInt(3.W))
    val s_axi_awburst_0 = Input(UInt(2.W))
    val s_axi_awvalid_0 = Input(Bool())
    val s_axi_awready_0 = Output(Bool())
    val s_axi_wdata_0   = Input(UInt(256.W))
    val s_axi_wstrb_0   = Input(UInt(32.W))
    val s_axi_wlast_0   = Input(Bool())
    val s_axi_wvalid_0  = Input(Bool())
    val s_axi_wready_0  = Output(Bool())
    val s_axi_bid_0     = Output(UInt(6.W))
    val s_axi_bresp_0   = Output(UInt(2.W))
    val s_axi_bvalid_0  = Output(Bool())
    val s_axi_bready_0  = Input(Bool())
    val s_axi_arid_0    = Input(UInt(6.W))
    val s_axi_araddr_0  = Input(UInt(33.W))
    val s_axi_arlen_0   = Input(UInt(8.W))
    val s_axi_arsize_0  = Input(UInt(3.W))
    val s_axi_arburst_0 = Input(UInt(2.W))
    val s_axi_arvalid_0 = Input(Bool())
    val s_axi_arready_0 = Output(Bool())
    val s_axi_rid_0     = Output(UInt(6.W))
    val s_axi_rdata_0   = Output(UInt(256.W))
    val s_axi_rresp_0   = Output(UInt(2.W))
    val s_axi_rlast_0   = Output(Bool())
    val s_axi_rvalid_0  = Output(Bool())
    val s_axi_rready_0  = Input(Bool())

    // Master port 1: ChipTop ExtMem AXI4
    val s_axi_awid_1    = Input(UInt(6.W))
    val s_axi_awaddr_1  = Input(UInt(33.W))
    val s_axi_awlen_1   = Input(UInt(8.W))
    val s_axi_awsize_1  = Input(UInt(3.W))
    val s_axi_awburst_1 = Input(UInt(2.W))
    val s_axi_awvalid_1 = Input(Bool())
    val s_axi_awready_1 = Output(Bool())
    val s_axi_wdata_1   = Input(UInt(256.W))
    val s_axi_wstrb_1   = Input(UInt(32.W))
    val s_axi_wlast_1   = Input(Bool())
    val s_axi_wvalid_1  = Input(Bool())
    val s_axi_wready_1  = Output(Bool())
    val s_axi_bid_1     = Output(UInt(6.W))
    val s_axi_bresp_1   = Output(UInt(2.W))
    val s_axi_bvalid_1  = Output(Bool())
    val s_axi_bready_1  = Input(Bool())
    val s_axi_arid_1    = Input(UInt(6.W))
    val s_axi_araddr_1  = Input(UInt(33.W))
    val s_axi_arlen_1   = Input(UInt(8.W))
    val s_axi_arsize_1  = Input(UInt(3.W))
    val s_axi_arburst_1 = Input(UInt(2.W))
    val s_axi_arvalid_1 = Input(Bool())
    val s_axi_arready_1 = Output(Bool())
    val s_axi_rid_1     = Output(UInt(6.W))
    val s_axi_rdata_1   = Output(UInt(256.W))
    val s_axi_rresp_1   = Output(UInt(2.W))
    val s_axi_rlast_1   = Output(Bool())
    val s_axi_rvalid_1  = Output(Bool())
    val s_axi_rready_1  = Input(Bool())

    // Slave port 0: HBM2 PC0
    val m_axi_awid_0    = Output(UInt(6.W))
    val m_axi_awaddr_0  = Output(UInt(33.W))
    val m_axi_awlen_0   = Output(UInt(8.W))
    val m_axi_awsize_0  = Output(UInt(3.W))
    val m_axi_awburst_0 = Output(UInt(2.W))
    val m_axi_awvalid_0 = Output(Bool())
    val m_axi_awready_0 = Input(Bool())
    val m_axi_wdata_0   = Output(UInt(256.W))
    val m_axi_wstrb_0   = Output(UInt(32.W))
    val m_axi_wlast_0   = Output(Bool())
    val m_axi_wvalid_0  = Output(Bool())
    val m_axi_wready_0  = Input(Bool())
    val m_axi_bid_0     = Input(UInt(6.W))
    val m_axi_bresp_0   = Input(UInt(2.W))
    val m_axi_bvalid_0  = Input(Bool())
    val m_axi_bready_0  = Output(Bool())
    val m_axi_arid_0    = Output(UInt(6.W))
    val m_axi_araddr_0  = Output(UInt(33.W))
    val m_axi_arlen_0   = Output(UInt(8.W))
    val m_axi_arsize_0  = Output(UInt(3.W))
    val m_axi_arburst_0 = Output(UInt(2.W))
    val m_axi_arvalid_0 = Output(Bool())
    val m_axi_arready_0 = Input(Bool())
    val m_axi_rid_0     = Input(UInt(6.W))
    val m_axi_rdata_0   = Input(UInt(256.W))
    val m_axi_rresp_0   = Input(UInt(2.W))
    val m_axi_rlast_0   = Input(Bool())
    val m_axi_rvalid_0  = Input(Bool())
    val m_axi_rready_0  = Output(Bool())
  })
}

// AXI Width Converter: 512-bit → 256-bit (for XDMA DMA → Crossbar)
class AXIWidthConverterBlackBox extends BlackBox {
  override def desiredName = "axi_dwidth_converter_0"
  val io = IO(new Bundle {
    val aclk    = Input(Clock())
    val aresetn = Input(Bool())

    // Slave side: 512-bit (from XDMA DMA)
    val s_axi_awid    = Input(UInt(4.W))
    val s_axi_awaddr  = Input(UInt(64.W))
    val s_axi_awlen   = Input(UInt(8.W))
    val s_axi_awsize  = Input(UInt(3.W))
    val s_axi_awburst = Input(UInt(2.W))
    val s_axi_awvalid = Input(Bool())
    val s_axi_awready = Output(Bool())
    val s_axi_wdata   = Input(UInt(512.W))
    val s_axi_wstrb   = Input(UInt(64.W))
    val s_axi_wlast   = Input(Bool())
    val s_axi_wvalid  = Input(Bool())
    val s_axi_wready  = Output(Bool())
    val s_axi_bid     = Output(UInt(4.W))
    val s_axi_bresp   = Output(UInt(2.W))
    val s_axi_bvalid  = Output(Bool())
    val s_axi_bready  = Input(Bool())
    val s_axi_arid    = Input(UInt(4.W))
    val s_axi_araddr  = Input(UInt(64.W))
    val s_axi_arlen   = Input(UInt(8.W))
    val s_axi_arsize  = Input(UInt(3.W))
    val s_axi_arburst = Input(UInt(2.W))
    val s_axi_arvalid = Input(Bool())
    val s_axi_arready = Output(Bool())
    val s_axi_rid     = Output(UInt(4.W))
    val s_axi_rdata   = Output(UInt(512.W))
    val s_axi_rresp   = Output(UInt(2.W))
    val s_axi_rlast   = Output(Bool())
    val s_axi_rvalid  = Output(Bool())
    val s_axi_rready  = Input(Bool())

    // Master side: 256-bit (to crossbar)
    val m_axi_awid    = Output(UInt(6.W))
    val m_axi_awaddr  = Output(UInt(33.W))
    val m_axi_awlen   = Output(UInt(8.W))
    val m_axi_awsize  = Output(UInt(3.W))
    val m_axi_awburst = Output(UInt(2.W))
    val m_axi_awvalid = Output(Bool())
    val m_axi_awready = Input(Bool())
    val m_axi_wdata   = Output(UInt(256.W))
    val m_axi_wstrb   = Output(UInt(32.W))
    val m_axi_wlast   = Output(Bool())
    val m_axi_wvalid  = Output(Bool())
    val m_axi_wready  = Input(Bool())
    val m_axi_bid     = Input(UInt(6.W))
    val m_axi_bresp   = Input(UInt(2.W))
    val m_axi_bvalid  = Input(Bool())
    val m_axi_bready  = Output(Bool())
    val m_axi_arid    = Output(UInt(6.W))
    val m_axi_araddr  = Output(UInt(33.W))
    val m_axi_arlen   = Output(UInt(8.W))
    val m_axi_arsize  = Output(UInt(3.W))
    val m_axi_arburst = Output(UInt(2.W))
    val m_axi_arvalid = Output(Bool())
    val m_axi_arready = Input(Bool())
    val m_axi_rid     = Input(UInt(6.W))
    val m_axi_rdata   = Input(UInt(256.W))
    val m_axi_rresp   = Input(UInt(2.W))
    val m_axi_rlast   = Input(Bool())
    val m_axi_rvalid  = Input(Bool())
    val m_axi_rready  = Output(Bool())
  })
}

// PegasusShell: top-level Verilog module for AU280 FPGA
//
// Connects:
//   XDMA IP  <→> SCU (AXI-Lite BAR0)
//   XDMA IP  <→> AXI Crossbar master 0 (DMA)
//   ChipTop  <→> AXI Crossbar master 1 (ExtMem)
//   Crossbar <→> HBM2 PC0 (slave)
//   ChipTop UART TX → UARTCapture → XDMA AXI-Stream C2H
//   SCU BUFGCE → ChipTop clock
//   SCU dut_reset → ChipTop reset
//
// The ChipTop is NOT instantiated here — it is connected via the PegasusHarness
// (Chipyard's LazyModule infrastructure). PegasusShell only contains the IP
// blackboxes and glue logic. The PegasusHarness wraps PegasusShell and adds ChipTop.
//
class PegasusShell extends Module {
  val io = IO(new Bundle {
    // PCIe physical pins (to AU280 edge connector)
    val pcie_sys_clk      = Input(Clock())
    val pcie_sys_clk_gt   = Input(Clock())
    val pcie_sys_rst_n    = Input(Bool())
    val pcie_exp_txp      = Output(UInt(16.W))
    val pcie_exp_txn      = Output(UInt(16.W))
    val pcie_exp_rxp      = Input(UInt(16.W))
    val pcie_exp_rxn      = Input(UInt(16.W))

    // HBM2 reference clock (100 MHz, from MMCM or board clock)
    val hbm_ref_clk       = Input(Clock())

    // Exported DUT clock and reset (to PegasusHarness for ChipTop)
    val dut_clk           = Output(Clock())
    val dut_reset         = Output(Bool())

    // ChipTop AXI4 memory interface (from PegasusHarness, ExtMem punchthrough)
    // 256-bit data, 34-bit address (for 4GB address space)
    val chip_mem_awid     = Input(UInt(6.W))
    val chip_mem_awaddr   = Input(UInt(33.W))
    val chip_mem_awlen    = Input(UInt(8.W))
    val chip_mem_awsize   = Input(UInt(3.W))
    val chip_mem_awburst  = Input(UInt(2.W))
    val chip_mem_awvalid  = Input(Bool())
    val chip_mem_awready  = Output(Bool())
    val chip_mem_wdata    = Input(UInt(256.W))
    val chip_mem_wstrb    = Input(UInt(32.W))
    val chip_mem_wlast    = Input(Bool())
    val chip_mem_wvalid   = Input(Bool())
    val chip_mem_wready   = Output(Bool())
    val chip_mem_bid      = Output(UInt(6.W))
    val chip_mem_bresp    = Output(UInt(2.W))
    val chip_mem_bvalid   = Output(Bool())
    val chip_mem_bready   = Input(Bool())
    val chip_mem_arid     = Input(UInt(6.W))
    val chip_mem_araddr   = Input(UInt(33.W))
    val chip_mem_arlen    = Input(UInt(8.W))
    val chip_mem_arsize   = Input(UInt(3.W))
    val chip_mem_arburst  = Input(UInt(2.W))
    val chip_mem_arvalid  = Input(Bool())
    val chip_mem_arready  = Output(Bool())
    val chip_mem_rid      = Output(UInt(6.W))
    val chip_mem_rdata    = Output(UInt(256.W))
    val chip_mem_rresp    = Output(UInt(2.W))
    val chip_mem_rlast    = Output(Bool())
    val chip_mem_rvalid   = Output(Bool())
    val chip_mem_rready   = Input(Bool())

    // ChipTop UART TX (serial output from DUT)
    val uart_tx           = Input(Bool())
  })

  // --- Instantiate XDMA ---
  val xdma = Module(new XDMABlackBox)
  xdma.io.sys_clk    := io.pcie_sys_clk
  xdma.io.sys_clk_gt := io.pcie_sys_clk_gt
  xdma.io.sys_rst_n  := io.pcie_sys_rst_n
  io.pcie_exp_txp    := xdma.io.pci_exp_txp
  io.pcie_exp_txn    := xdma.io.pci_exp_txn
  xdma.io.pci_exp_rxp := io.pcie_exp_rxp
  xdma.io.pci_exp_rxn := io.pcie_exp_rxn

  val axiClk    = xdma.io.axi_aclk
  val axiResetn = xdma.io.axi_aresetn

  // --- Instantiate SCU (runs on host AXI clock) ---
  // The SCU module uses the implicit clock from withClockAndReset context.
  // We need it to run on axiClk.
  val scu = withClockAndReset(axiClk, !axiResetn) {
    Module(new SCU)
  }
  scu.io.host_clk := axiClk

  // Connect XDMA AXI-Lite master to SCU
  scu.io.axil.awvalid := xdma.io.m_axil_awvalid
  xdma.io.m_axil_awready := scu.io.axil.awready
  scu.io.axil.awaddr  := xdma.io.m_axil_awaddr
  scu.io.axil.awprot  := xdma.io.m_axil_awprot
  scu.io.axil.wvalid  := xdma.io.m_axil_wvalid
  xdma.io.m_axil_wready  := scu.io.axil.wready
  scu.io.axil.wdata   := xdma.io.m_axil_wdata
  scu.io.axil.wstrb   := xdma.io.m_axil_wstrb
  xdma.io.m_axil_bvalid  := scu.io.axil.bvalid
  scu.io.axil.bready  := xdma.io.m_axil_bready
  xdma.io.m_axil_bresp   := scu.io.axil.bresp
  scu.io.axil.arvalid := xdma.io.m_axil_arvalid
  xdma.io.m_axil_arready := scu.io.axil.arready
  scu.io.axil.araddr  := xdma.io.m_axil_araddr
  scu.io.axil.arprot  := xdma.io.m_axil_arprot
  xdma.io.m_axil_rvalid  := scu.io.axil.rvalid
  scu.io.axil.rready  := xdma.io.m_axil_rready
  xdma.io.m_axil_rdata   := scu.io.axil.rdata
  xdma.io.m_axil_rresp   := scu.io.axil.rresp

  // Export DUT clock and reset
  io.dut_clk   := scu.io.dut_clk
  io.dut_reset := scu.io.dut_reset

  // --- Instantiate UARTCapture ---
  val uartCapture = withClockAndReset(axiClk, !axiResetn) {
    Module(new UARTCapture())
  }
  uartCapture.io.uart_tx := io.uart_tx

  // Connect UART FIFO to XDMA AXI-Stream C2H channel 0
  // XDMA C2H channel expects 512-bit wide tdata; pad the 8-bit byte
  xdma.io.s_axis_c2h_tvalid_0 := uartCapture.io.axis_tvalid
  uartCapture.io.axis_tready  := xdma.io.s_axis_c2h_tready_0
  xdma.io.s_axis_c2h_tdata_0  := Cat(0.U(504.W), uartCapture.io.axis_tdata)
  xdma.io.s_axis_c2h_tlast_0  := uartCapture.io.axis_tlast
  xdma.io.s_axis_c2h_tkeep_0  := Cat(0.U(63.W), uartCapture.io.axis_tkeep)

  // --- Instantiate AXI Width Converter (XDMA 512-bit → 256-bit) ---
  val widthConv = Module(new AXIWidthConverterBlackBox)
  widthConv.io.aclk    := axiClk
  widthConv.io.aresetn := axiResetn

  // Connect XDMA DMA AXI4 master to width converter
  widthConv.io.s_axi_awid    := xdma.io.m_axi_awid
  widthConv.io.s_axi_awaddr  := xdma.io.m_axi_awaddr
  widthConv.io.s_axi_awlen   := xdma.io.m_axi_awlen
  widthConv.io.s_axi_awsize  := xdma.io.m_axi_awsize
  widthConv.io.s_axi_awburst := xdma.io.m_axi_awburst
  widthConv.io.s_axi_awvalid := xdma.io.m_axi_awvalid
  xdma.io.m_axi_awready      := widthConv.io.s_axi_awready
  widthConv.io.s_axi_wdata   := xdma.io.m_axi_wdata
  widthConv.io.s_axi_wstrb   := xdma.io.m_axi_wstrb
  widthConv.io.s_axi_wlast   := xdma.io.m_axi_wlast
  widthConv.io.s_axi_wvalid  := xdma.io.m_axi_wvalid
  xdma.io.m_axi_wready       := widthConv.io.s_axi_wready
  xdma.io.m_axi_bid          := widthConv.io.s_axi_bid
  xdma.io.m_axi_bresp        := widthConv.io.s_axi_bresp
  xdma.io.m_axi_bvalid       := widthConv.io.s_axi_bvalid
  widthConv.io.s_axi_bready  := xdma.io.m_axi_bready
  widthConv.io.s_axi_arid    := xdma.io.m_axi_arid
  widthConv.io.s_axi_araddr  := xdma.io.m_axi_araddr
  widthConv.io.s_axi_arlen   := xdma.io.m_axi_arlen
  widthConv.io.s_axi_arsize  := xdma.io.m_axi_arsize
  widthConv.io.s_axi_arburst := xdma.io.m_axi_arburst
  widthConv.io.s_axi_arvalid := xdma.io.m_axi_arvalid
  xdma.io.m_axi_arready      := widthConv.io.s_axi_arready
  xdma.io.m_axi_rid          := widthConv.io.s_axi_rid
  xdma.io.m_axi_rdata        := widthConv.io.s_axi_rdata
  xdma.io.m_axi_rresp        := widthConv.io.s_axi_rresp
  xdma.io.m_axi_rlast        := widthConv.io.s_axi_rlast
  xdma.io.m_axi_rvalid       := widthConv.io.s_axi_rvalid
  widthConv.io.s_axi_rready  := xdma.io.m_axi_rready

  // --- Instantiate AXI Crossbar ---
  val crossbar = Module(new AXICrossbarBlackBox)
  crossbar.io.aclk    := axiClk
  crossbar.io.aresetn := axiResetn

  // Connect width converter output to crossbar master port 0 (XDMA DMA)
  crossbar.io.s_axi_awid_0    := widthConv.io.m_axi_awid
  crossbar.io.s_axi_awaddr_0  := widthConv.io.m_axi_awaddr
  crossbar.io.s_axi_awlen_0   := widthConv.io.m_axi_awlen
  crossbar.io.s_axi_awsize_0  := widthConv.io.m_axi_awsize
  crossbar.io.s_axi_awburst_0 := widthConv.io.m_axi_awburst
  crossbar.io.s_axi_awvalid_0 := widthConv.io.m_axi_awvalid
  widthConv.io.m_axi_awready  := crossbar.io.s_axi_awready_0
  crossbar.io.s_axi_wdata_0   := widthConv.io.m_axi_wdata
  crossbar.io.s_axi_wstrb_0   := widthConv.io.m_axi_wstrb
  crossbar.io.s_axi_wlast_0   := widthConv.io.m_axi_wlast
  crossbar.io.s_axi_wvalid_0  := widthConv.io.m_axi_wvalid
  widthConv.io.m_axi_wready   := crossbar.io.s_axi_wready_0
  widthConv.io.m_axi_bid      := crossbar.io.s_axi_bid_0
  widthConv.io.m_axi_bresp    := crossbar.io.s_axi_bresp_0
  widthConv.io.m_axi_bvalid   := crossbar.io.s_axi_bvalid_0
  crossbar.io.s_axi_bready_0  := widthConv.io.m_axi_bready
  crossbar.io.s_axi_arid_0    := widthConv.io.m_axi_arid
  crossbar.io.s_axi_araddr_0  := widthConv.io.m_axi_araddr
  crossbar.io.s_axi_arlen_0   := widthConv.io.m_axi_arlen
  crossbar.io.s_axi_arsize_0  := widthConv.io.m_axi_arsize
  crossbar.io.s_axi_arburst_0 := widthConv.io.m_axi_arburst
  crossbar.io.s_axi_arvalid_0 := widthConv.io.m_axi_arvalid
  widthConv.io.m_axi_arready  := crossbar.io.s_axi_arready_0
  widthConv.io.m_axi_rid      := crossbar.io.s_axi_rid_0
  widthConv.io.m_axi_rdata    := crossbar.io.s_axi_rdata_0
  widthConv.io.m_axi_rresp    := crossbar.io.s_axi_rresp_0
  widthConv.io.m_axi_rlast    := crossbar.io.s_axi_rlast_0
  widthConv.io.m_axi_rvalid   := crossbar.io.s_axi_rvalid_0
  crossbar.io.s_axi_rready_0  := widthConv.io.m_axi_rready

  // Connect ChipTop ExtMem to crossbar master port 1
  // Address offset: SoC physical addr 0x80000000 → HBM2 addr 0x0
  // The address translation is done here: chip_mem_araddr - 0x80000000
  crossbar.io.s_axi_awid_1    := io.chip_mem_awid
  crossbar.io.s_axi_awaddr_1  := io.chip_mem_awaddr - 0x80000000L.U(33.W)
  crossbar.io.s_axi_awlen_1   := io.chip_mem_awlen
  crossbar.io.s_axi_awsize_1  := io.chip_mem_awsize
  crossbar.io.s_axi_awburst_1 := io.chip_mem_awburst
  crossbar.io.s_axi_awvalid_1 := io.chip_mem_awvalid
  io.chip_mem_awready          := crossbar.io.s_axi_awready_1
  crossbar.io.s_axi_wdata_1   := io.chip_mem_wdata
  crossbar.io.s_axi_wstrb_1   := io.chip_mem_wstrb
  crossbar.io.s_axi_wlast_1   := io.chip_mem_wlast
  crossbar.io.s_axi_wvalid_1  := io.chip_mem_wvalid
  io.chip_mem_wready           := crossbar.io.s_axi_wready_1
  io.chip_mem_bid              := crossbar.io.s_axi_bid_1
  io.chip_mem_bresp            := crossbar.io.s_axi_bresp_1
  io.chip_mem_bvalid           := crossbar.io.s_axi_bvalid_1
  crossbar.io.s_axi_bready_1  := io.chip_mem_bready
  crossbar.io.s_axi_arid_1    := io.chip_mem_arid
  crossbar.io.s_axi_araddr_1  := io.chip_mem_araddr - 0x80000000L.U(33.W)
  crossbar.io.s_axi_arlen_1   := io.chip_mem_arlen
  crossbar.io.s_axi_arsize_1  := io.chip_mem_arsize
  crossbar.io.s_axi_arburst_1 := io.chip_mem_arburst
  crossbar.io.s_axi_arvalid_1 := io.chip_mem_arvalid
  io.chip_mem_arready          := crossbar.io.s_axi_arready_1
  io.chip_mem_rid              := crossbar.io.s_axi_rid_1
  io.chip_mem_rdata            := crossbar.io.s_axi_rdata_1
  io.chip_mem_rresp            := crossbar.io.s_axi_rresp_1
  io.chip_mem_rlast            := crossbar.io.s_axi_rlast_1
  io.chip_mem_rvalid           := crossbar.io.s_axi_rvalid_1
  crossbar.io.s_axi_rready_1  := io.chip_mem_rready

  // --- Instantiate HBM2 ---
  val hbm2 = Module(new HBM2BlackBox)
  hbm2.io.HBM_REF_CLK_0   := io.hbm_ref_clk
  hbm2.io.AXI_00_ACLK     := axiClk
  hbm2.io.AXI_00_ARESET_N := axiResetn

  // Connect crossbar slave port 0 to HBM2 PC0
  hbm2.io.AXI_00_AWID     := crossbar.io.m_axi_awid_0
  hbm2.io.AXI_00_AWADDR   := crossbar.io.m_axi_awaddr_0
  hbm2.io.AXI_00_AWLEN    := crossbar.io.m_axi_awlen_0(3, 0)  // HBM2 only has 4-bit len
  hbm2.io.AXI_00_AWSIZE   := crossbar.io.m_axi_awsize_0
  hbm2.io.AXI_00_AWBURST  := crossbar.io.m_axi_awburst_0
  hbm2.io.AXI_00_AWVALID  := crossbar.io.m_axi_awvalid_0
  crossbar.io.m_axi_awready_0 := hbm2.io.AXI_00_AWREADY
  hbm2.io.AXI_00_WDATA    := crossbar.io.m_axi_wdata_0
  hbm2.io.AXI_00_WSTRB    := crossbar.io.m_axi_wstrb_0
  hbm2.io.AXI_00_WLAST    := crossbar.io.m_axi_wlast_0
  hbm2.io.AXI_00_WVALID   := crossbar.io.m_axi_wvalid_0
  crossbar.io.m_axi_wready_0 := hbm2.io.AXI_00_WREADY
  crossbar.io.m_axi_bid_0   := hbm2.io.AXI_00_BID
  crossbar.io.m_axi_bresp_0 := hbm2.io.AXI_00_BRESP
  crossbar.io.m_axi_bvalid_0 := hbm2.io.AXI_00_BVALID
  hbm2.io.AXI_00_BREADY   := crossbar.io.m_axi_bready_0
  hbm2.io.AXI_00_ARID     := crossbar.io.m_axi_arid_0
  hbm2.io.AXI_00_ARADDR   := crossbar.io.m_axi_araddr_0
  hbm2.io.AXI_00_ARLEN    := crossbar.io.m_axi_arlen_0(3, 0)
  hbm2.io.AXI_00_ARSIZE   := crossbar.io.m_axi_arsize_0
  hbm2.io.AXI_00_ARBURST  := crossbar.io.m_axi_arburst_0
  hbm2.io.AXI_00_ARVALID  := crossbar.io.m_axi_arvalid_0
  crossbar.io.m_axi_arready_0 := hbm2.io.AXI_00_ARREADY
  crossbar.io.m_axi_rid_0   := hbm2.io.AXI_00_RID
  crossbar.io.m_axi_rdata_0 := hbm2.io.AXI_00_RDATA
  crossbar.io.m_axi_rresp_0 := hbm2.io.AXI_00_RRESP
  crossbar.io.m_axi_rlast_0 := hbm2.io.AXI_00_RLAST
  crossbar.io.m_axi_rvalid_0 := hbm2.io.AXI_00_RVALID
  hbm2.io.AXI_00_RREADY   := crossbar.io.m_axi_rready_0

  // HBM2 APB configuration interface: tie off (no runtime config needed)
  hbm2.io.APB_0_PWDATA   := 0.U
  hbm2.io.APB_0_PADDR    := 0.U
  hbm2.io.APB_0_PCLK     := axiClk
  hbm2.io.APB_0_PENABLE  := false.B
  hbm2.io.APB_0_PRESET_N := axiResetn
  hbm2.io.APB_0_PSEL     := false.B
  hbm2.io.APB_0_PWRITE   := false.B
}
