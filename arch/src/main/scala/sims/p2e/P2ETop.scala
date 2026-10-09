package sims.p2e

import chisel3._
import chisel3.experimental.Analog
import chisel3.util._
import memcore.bus.axi4
import memcore.memory.queue.Queue
import sims.soc.{SimSoc, SystemTarget}

/** A chip on the P2E board: the System over the 16-GiB DDR4 macro at 0x80000000. */
abstract class P2ETarget(pb: String) extends SystemTarget(pb) {
  override val dramBytes: BigInt = BigInt(16) << 30
}

/** Board pins of the P2E top; bebop's Tcl drives `sys_rstn`/`soc_hold` and reads the status pins. */
class P2ETopIO extends Bundle {
  val user_clk = Input(Clock())
  val sys_rstn = Input(Bool())
  // Holds the SoC in reset after DDR calibration until the host has loaded every image.
  val soc_hold = Input(Bool())

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

/** AXI slave port of the DDR4 macro, as named on the macro (`<prefix>_ddr4_s_axi_*`). */
class Ddr4AxiSlave(idBits: Int) extends Bundle {
  val awid    = Input(UInt(idBits.W))
  val awaddr  = Input(UInt(64.W))
  val awlen   = Input(UInt(8.W))
  val awsize  = Input(UInt(3.W))
  val awburst = Input(UInt(2.W))
  val awlock  = Input(UInt(1.W))
  val awcache = Input(UInt(4.W))
  val awprot  = Input(UInt(3.W))
  val awqos   = Input(UInt(4.W))
  val awvalid = Input(Bool())
  val awready = Output(Bool())
  val wdata   = Input(UInt(256.W))
  val wstrb   = Input(UInt(32.W))
  val wlast   = Input(Bool())
  val wvalid  = Input(Bool())
  val wready  = Output(Bool())
  val bready  = Input(Bool())
  val bid     = Output(UInt(idBits.W))
  val bresp   = Output(UInt(2.W))
  val bvalid  = Output(Bool())
  val arid    = Input(UInt(idBits.W))
  val araddr  = Input(UInt(64.W))
  val arlen   = Input(UInt(8.W))
  val arsize  = Input(UInt(3.W))
  val arburst = Input(UInt(2.W))
  val arlock  = Input(UInt(1.W))
  val arcache = Input(UInt(4.W))
  val arprot  = Input(UInt(3.W))
  val arqos   = Input(UInt(4.W))
  val arvalid = Input(Bool())
  val arready = Output(Bool())
  val rready  = Input(Bool())
  val rid     = Output(UInt(idBits.W))
  val rdata   = Output(UInt(256.W))
  val rresp   = Output(UInt(2.W))
  val rlast   = Output(Bool())
  val rvalid  = Output(Bool())
}

/** The board DDR4 controller; vcom replaces this stub with the `xepic_ddr4_dc1` netlist macro. */
class XepicDdr4Dc1 extends BlackBox {
  override def desiredName = "xepic_ddr4_dc1"

  val io = IO(new Bundle {
    val sys_rstn               = Input(Bool())
    val c0_sys_clk_p           = Input(Bool())
    val c0_sys_clk_n           = Input(Bool())
    val c0_ddr4_act_n          = Output(Bool())
    val c0_ddr4_adr            = Output(UInt(17.W))
    val c0_ddr4_ba             = Output(UInt(2.W))
    val c0_ddr4_bg             = Output(UInt(2.W))
    val c0_ddr4_cke            = Output(UInt(2.W))
    val c0_ddr4_odt            = Output(UInt(2.W))
    val c0_ddr4_cs_n           = Output(UInt(2.W))
    val c0_ddr4_ck_t           = Output(UInt(2.W))
    val c0_ddr4_ck_c           = Output(UInt(2.W))
    val c0_ddr4_reset_n        = Output(Bool())
    val c0_ddr4_dm_dbi_n       = Analog(8.W)
    val c0_ddr4_dq             = Analog(64.W)
    val c0_ddr4_dqs_c          = Analog(8.W)
    val c0_ddr4_dqs_t          = Analog(8.W)
    val gclk_100m              = Input(Bool())
    val ddr4_en_vtt_bbox       = Output(Bool())
    val ddr4_en_vddq_bbox      = Output(Bool())
    val ddr4_en_vcc2v5_bbox    = Output(Bool())
    val power_good_bbox        = Input(Bool())
    val init_start             = Input(Bool())
    val init_cfg               = Input(Bool())
    val init_busy              = Output(Bool())
    val init_calib_complete    = Output(Bool())
    val c0_init_calib_complete = Output(Bool())
    val axi_clk                = Input(Clock())
    val c0_ddr4_ui_clk         = Output(Clock())
    // Flattened as the macro's `s0_ddr4_s_axi_*` and `s1_ddr4_s_axi_*` ports.
    val s0_ddr4_s_axi          = new Ddr4AxiSlave(11)
    val s1_ddr4_s_axi          = new Ddr4AxiSlave(4)
  })

}

/**
 * Maps the System's 128-bit DDR AXI onto the macro's 256-bit slave. Transfers stay 16-byte
 * narrow bursts, so only lanes move: beat k of a burst starting at A uses lane (A + 16k)[4].
 * CPU 0x80000000 is DDR offset 0, and the System's IDs widen to the macro's 11 bits.
 */
class P2EDdrAdapter(system: axi4.Params, base: BigInt) extends Module {
  require(system.dataBits == 128, "P2E DDR adapter maps 16-byte beats onto the 32-byte macro bus")
  require(system.idBits <= 11)

  val io = IO(new Bundle {
    val in  = Flipped(new axi4.Port(system))
    val out = Flipped(new Ddr4AxiSlave(11))
  })

  val lanes = 1 << system.idBits

  // Write lanes: W beats follow AW order, so queue each accepted burst's first lane.
  val writeLane  = Module(new Queue(Bool(), 8))
  val writeOdd   = RegInit(false.B)
  val writeUpper = writeLane.io.deq.bits ^ writeOdd
  io.out.awid            := io.in.aw.bits.id
  io.out.awaddr          := io.in.aw.bits.addr.pad(64) - base.U(64.W)
  io.out.awlen           := io.in.aw.bits.len
  io.out.awsize          := io.in.aw.bits.size
  io.out.awburst         := io.in.aw.bits.burst
  io.out.awlock          := 0.U
  io.out.awcache         := "b0011".U
  io.out.awprot          := 0.U
  io.out.awqos           := 0.U
  io.out.awvalid         := io.in.aw.valid && writeLane.io.enq.ready
  io.in.aw.ready         := io.out.awready && writeLane.io.enq.ready
  writeLane.io.enq.valid := io.in.aw.fire
  writeLane.io.enq.bits  := io.in.aw.bits.addr(4)
  when(io.in.aw.fire) {
    assert(
      io.in.aw.bits.size === 4.U && io.in.aw.bits.addr(3, 0) === 0.U && io.in.aw.bits.burst === 1.U,
      "P2E DDR adapter expects aligned 16-byte INCR bursts"
    )
  }

  io.out.wdata                := Fill(2, io.in.w.bits.data)
  io.out.wstrb                := Mux(writeUpper, io.in.w.bits.strb ## 0.U(16.W), 0.U(16.W) ## io.in.w.bits.strb)
  io.out.wlast                := io.in.w.bits.last
  io.out.wvalid               := io.in.w.valid && writeLane.io.deq.valid
  io.in.w.ready               := io.out.wready && writeLane.io.deq.valid
  writeLane.io.deq.ready      := io.in.w.fire && io.in.w.bits.last
  when(io.in.w.fire)(writeOdd := Mux(io.in.w.bits.last, false.B, !writeOdd))

  io.in.b.valid     := io.out.bvalid
  io.in.b.bits.id   := io.out.bid(system.idBits - 1, 0)
  io.in.b.bits.resp := io.out.bresp
  io.out.bready     := io.in.b.ready

  // Read lanes: a System ID has at most one read in flight, so one lane bit per ID suffices.
  val readBusy  = RegInit(VecInit(Seq.fill(lanes)(false.B)))
  val readUpper = Reg(Vec(lanes, Bool()))
  io.out.arid    := io.in.ar.bits.id
  io.out.araddr  := io.in.ar.bits.addr.pad(64) - base.U(64.W)
  io.out.arlen   := io.in.ar.bits.len
  io.out.arsize  := io.in.ar.bits.size
  io.out.arburst := io.in.ar.bits.burst
  io.out.arlock  := 0.U
  io.out.arcache := "b0011".U
  io.out.arprot  := 0.U
  io.out.arqos   := 0.U
  io.out.arvalid := io.in.ar.valid
  io.in.ar.ready := io.out.arready
  when(io.in.ar.fire) {
    assert(
      io.in.ar.bits.size === 4.U && io.in.ar.bits.addr(3, 0) === 0.U && io.in.ar.bits.burst === 1.U,
      "P2E DDR adapter expects aligned 16-byte INCR bursts"
    )
    assert(!readBusy(io.in.ar.bits.id), "P2E DDR adapter allows one read in flight per ID")
    readBusy(io.in.ar.bits.id)  := true.B
    readUpper(io.in.ar.bits.id) := io.in.ar.bits.addr(4)
  }

  val rid = io.out.rid(system.idBits - 1, 0)
  io.in.r.valid     := io.out.rvalid
  io.in.r.bits.id   := rid
  io.in.r.bits.data := Mux(readUpper(rid), io.out.rdata(255, 128), io.out.rdata(127, 0))
  io.in.r.bits.resp := io.out.rresp
  io.in.r.bits.last := io.out.rlast
  io.out.rready     := io.in.r.ready
  when(io.in.r.fire) {
    readUpper(rid)                        := !readUpper(rid)
    when(io.in.r.bits.last)(readBusy(rid) := false.B)
  }
}

/**
 * The `top` instance bebop's flow addresses (`P2ETop.top.ddr`, `P2ETop.top.user_clk`): DDR4 macro,
 * DDR adapter and the System, all on `user_clk`. The SoC leaves reset only after calibration and
 * after the host releases `soc_hold`.
 */
class P2EHarness(target: SystemTarget, diffTest: Boolean) extends RawModule {
  val io = FlatIO(new P2ETopIO)

  val ddr = Module(new XepicDdr4Dc1)
  ddr.io.sys_rstn        := io.sys_rstn
  ddr.io.c0_sys_clk_p    := io.c0_sys_clk_p
  ddr.io.c0_sys_clk_n    := io.c0_sys_clk_n
  io.c0_ddr4_act_n       := ddr.io.c0_ddr4_act_n
  io.c0_ddr4_adr         := ddr.io.c0_ddr4_adr
  io.c0_ddr4_ba          := ddr.io.c0_ddr4_ba
  io.c0_ddr4_bg          := ddr.io.c0_ddr4_bg
  io.c0_ddr4_cke         := ddr.io.c0_ddr4_cke
  io.c0_ddr4_odt         := ddr.io.c0_ddr4_odt
  io.c0_ddr4_cs_n        := ddr.io.c0_ddr4_cs_n
  io.c0_ddr4_ck_t        := ddr.io.c0_ddr4_ck_t
  io.c0_ddr4_ck_c        := ddr.io.c0_ddr4_ck_c
  io.c0_ddr4_reset_n     := ddr.io.c0_ddr4_reset_n
  ddr.io.c0_ddr4_dm_dbi_n <> io.c0_ddr4_dm_dbi_n
  ddr.io.c0_ddr4_dq <> io.c0_ddr4_dq
  ddr.io.c0_ddr4_dqs_c <> io.c0_ddr4_dqs_c
  ddr.io.c0_ddr4_dqs_t <> io.c0_ddr4_dqs_t
  ddr.io.gclk_100m       := false.B
  io.ddr4_en_vtt         := ddr.io.ddr4_en_vtt_bbox
  io.ddr4_en_vddq        := ddr.io.ddr4_en_vddq_bbox
  io.ddr4_en_vcc2v5      := ddr.io.ddr4_en_vcc2v5_bbox
  ddr.io.power_good_bbox := io.power_good
  ddr.io.init_start      := true.B
  ddr.io.init_cfg        := false.B
  ddr.io.axi_clk         := io.user_clk
  io.c0_ddr4_ui_clk      := ddr.io.c0_ddr4_ui_clk
  io.init_calib_complete := ddr.io.c0_init_calib_complete

  val socReset = !io.sys_rstn || !ddr.io.c0_init_calib_complete || io.soc_hold

  withClockAndReset(io.user_clk, socReset) {
    val soc     = Module(new SimSoc(target, diffTest))
    val adapter = Module(new P2EDdrAdapter(soc.axiParams, target.dramBase))
    adapter.io.in <> soc.io.axi
    ddr.io.s0_ddr4_s_axi <> adapter.io.out
  }

  // The second macro port is unused.
  ddr.io.s1_ddr4_s_axi <> DontCare
  ddr.io.s1_ddr4_s_axi.awvalid := false.B
  ddr.io.s1_ddr4_s_axi.wvalid  := false.B
  ddr.io.s1_ddr4_s_axi.arvalid := false.B
  ddr.io.s1_ddr4_s_axi.bready  := true.B
  ddr.io.s1_ddr4_s_axi.rready  := true.B
}

/** The P2E top module bebop's vvac/vcom flow builds; `P2ETop.top` is the harness. */
class P2ETop(target: SystemTarget, diffTest: Boolean) extends RawModule {
  val io  = IO(new P2ETopIO)
  val top = Module(new P2EHarness(target, diffTest))
  // The DDR4 data pins stay on the macro inside `top`: vcom binds the macro's own pins, and a top
  // inout driving the macro is rejected.
  for ((name, port) <- io.elements) port match {
    case _: Analog =>
    case _ => top.io.elements(name) <> port
  }
}
