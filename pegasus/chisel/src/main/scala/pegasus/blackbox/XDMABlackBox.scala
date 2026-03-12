package pegasus.blackbox

import chisel3._
import chisel3.util._

// AXI4-Lite bundle for XDMA AXI-Lite master interface
class AXILiteBundle(addrWidth: Int = 32, dataWidth: Int = 32) extends Bundle {
  val awvalid = Output(Bool())
  val awready = Input(Bool())
  val awaddr  = Output(UInt(addrWidth.W))
  val awprot  = Output(UInt(3.W))

  val wvalid  = Output(Bool())
  val wready  = Input(Bool())
  val wdata   = Output(UInt(dataWidth.W))
  val wstrb   = Output(UInt((dataWidth / 8).W))

  val bvalid  = Input(Bool())
  val bready  = Output(Bool())
  val bresp   = Input(UInt(2.W))

  val arvalid = Output(Bool())
  val arready = Input(Bool())
  val araddr  = Output(UInt(addrWidth.W))
  val arprot  = Output(UInt(3.W))

  val rvalid  = Input(Bool())
  val rready  = Output(Bool())
  val rdata   = Input(UInt(dataWidth.W))
  val rresp   = Input(UInt(2.W))
}

// AXI4 bundle for XDMA DMA master interface (connects to HBM2 via crossbar)
class AXI4Bundle(
  addrWidth: Int = 64,
  dataWidth: Int = 512,
  idWidth:   Int = 4
) extends Bundle {
  val awid    = Output(UInt(idWidth.W))
  val awaddr  = Output(UInt(addrWidth.W))
  val awlen   = Output(UInt(8.W))
  val awsize  = Output(UInt(3.W))
  val awburst = Output(UInt(2.W))
  val awvalid = Output(Bool())
  val awready = Input(Bool())

  val wdata   = Output(UInt(dataWidth.W))
  val wstrb   = Output(UInt((dataWidth / 8).W))
  val wlast   = Output(Bool())
  val wvalid  = Output(Bool())
  val wready  = Input(Bool())

  val bid     = Input(UInt(idWidth.W))
  val bresp   = Input(UInt(2.W))
  val bvalid  = Input(Bool())
  val bready  = Output(Bool())

  val arid    = Output(UInt(idWidth.W))
  val araddr  = Output(UInt(addrWidth.W))
  val arlen   = Output(UInt(8.W))
  val arsize  = Output(UInt(3.W))
  val arburst = Output(UInt(2.W))
  val arvalid = Output(Bool())
  val arready = Input(Bool())

  val rid     = Input(UInt(idWidth.W))
  val rdata   = Input(UInt(dataWidth.W))
  val rresp   = Input(UInt(2.W))
  val rlast   = Input(Bool())
  val rvalid  = Input(Bool())
  val rready  = Output(Bool())
}

// AXI-Stream bundle (XDMA C2H channel for UART)
class AXIStreamBundle(dataWidth: Int = 8) extends Bundle {
  val tvalid = Input(Bool())
  val tready = Output(Bool())
  val tdata  = Input(UInt(dataWidth.W))
  val tlast  = Input(Bool())
  val tkeep  = Input(UInt((dataWidth / 8).W))
}

// Xilinx XDMA IP black box
// PCIe x16 Gen3, 512-bit AXI, 250 MHz AXI clock
// Exposes:
//   - axi_aclk / axi_aresetn  : AXI clock/reset outputs (driven by IP)
//   - m_axil_*                : AXI-Lite master (for SCU MMIO)
//   - m_axi_*                 : AXI4 master (DMA, 512-bit, to HBM2 via crossbar)
//   - s_axis_c2h_*            : AXI-Stream slave (C2H, for UART upload)
//   - pcie_*                  : PCIe PHY pins
class XDMABlackBox extends BlackBox {
  override def desiredName = "xdma_0"
  val io = IO(new Bundle {
    // PCIe reference clock (100 MHz differential)
    val sys_clk   = Input(Clock())
    val sys_clk_gt = Input(Clock())
    val sys_rst_n  = Input(Bool())

    // Clocks/reset outputs (driven by XDMA IP)
    val axi_aclk    = Output(Clock())
    val axi_aresetn = Output(Bool())

    // AXI-Lite master (MMIO, BAR0)
    val m_axil_awvalid = Output(Bool())
    val m_axil_awready = Input(Bool())
    val m_axil_awaddr  = Output(UInt(32.W))
    val m_axil_awprot  = Output(UInt(3.W))
    val m_axil_wvalid  = Output(Bool())
    val m_axil_wready  = Input(Bool())
    val m_axil_wdata   = Output(UInt(32.W))
    val m_axil_wstrb   = Output(UInt(4.W))
    val m_axil_bvalid  = Input(Bool())
    val m_axil_bready  = Output(Bool())
    val m_axil_bresp   = Input(UInt(2.W))
    val m_axil_arvalid = Output(Bool())
    val m_axil_arready = Input(Bool())
    val m_axil_araddr  = Output(UInt(32.W))
    val m_axil_arprot  = Output(UInt(3.W))
    val m_axil_rvalid  = Input(Bool())
    val m_axil_rready  = Output(Bool())
    val m_axil_rdata   = Input(UInt(32.W))
    val m_axil_rresp   = Input(UInt(2.W))

    // AXI4 master (DMA, 512-bit data bus)
    val m_axi_awid    = Output(UInt(4.W))
    val m_axi_awaddr  = Output(UInt(64.W))
    val m_axi_awlen   = Output(UInt(8.W))
    val m_axi_awsize  = Output(UInt(3.W))
    val m_axi_awburst = Output(UInt(2.W))
    val m_axi_awprot  = Output(UInt(3.W))
    val m_axi_awvalid = Output(Bool())
    val m_axi_awready = Input(Bool())
    val m_axi_wdata   = Output(UInt(512.W))
    val m_axi_wstrb   = Output(UInt(64.W))
    val m_axi_wlast   = Output(Bool())
    val m_axi_wvalid  = Output(Bool())
    val m_axi_wready  = Input(Bool())
    val m_axi_bid     = Input(UInt(4.W))
    val m_axi_bresp   = Input(UInt(2.W))
    val m_axi_bvalid  = Input(Bool())
    val m_axi_bready  = Output(Bool())
    val m_axi_arid    = Output(UInt(4.W))
    val m_axi_araddr  = Output(UInt(64.W))
    val m_axi_arlen   = Output(UInt(8.W))
    val m_axi_arsize  = Output(UInt(3.W))
    val m_axi_arburst = Output(UInt(2.W))
    val m_axi_arprot  = Output(UInt(3.W))
    val m_axi_arvalid = Output(Bool())
    val m_axi_arready = Input(Bool())
    val m_axi_rid     = Input(UInt(4.W))
    val m_axi_rdata   = Input(UInt(512.W))
    val m_axi_rresp   = Input(UInt(2.W))
    val m_axi_rlast   = Input(Bool())
    val m_axi_rvalid  = Input(Bool())
    val m_axi_rready  = Output(Bool())

    // AXI-Stream slave C2H channel 0 (host-bound, e.g. UART output)
    val s_axis_c2h_tvalid_0 = Input(Bool())
    val s_axis_c2h_tready_0 = Output(Bool())
    val s_axis_c2h_tdata_0  = Input(UInt(512.W))
    val s_axis_c2h_tlast_0  = Input(Bool())
    val s_axis_c2h_tkeep_0  = Input(UInt(64.W))

    // PCIe differential pairs (x16)
    val pci_exp_txp = Output(UInt(16.W))
    val pci_exp_txn = Output(UInt(16.W))
    val pci_exp_rxp = Input(UInt(16.W))
    val pci_exp_rxn = Input(UInt(16.W))
  })
}
