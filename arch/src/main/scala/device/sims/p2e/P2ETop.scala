package framework.device.sims.p2e

import chisel3.experimental.hierarchy.public

import chisel3.experimental.hierarchy.instantiable

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.device.sims.configs.GlobalConfig

import chisel3._
import chisel3.experimental.Analog
import chisel3.util._
import framework.generic.bus.axi4
import framework.generic.queue.Queue
import framework.device.sims.soc.{SimSoc, SystemTarget}

/** A design on the P2E board: the System over the 16-GiB DDR4 macro at 0x80000000. */
abstract class P2ETarget(b: GlobalConfig) extends SystemTarget(b)

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
class Ddr4AxiSlave(port: Int) extends Bundle {
  val idBits = XepicDdr4Dc1.portIdBits(port)

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

object XepicDdr4Dc1 { val portIdBits = Seq(11, 4) }

/** The board DDR4 controller; vcom replaces this stub with the `xepic_ddr4_dc1` netlist macro. */
@instantiable
class XepicDdr4Dc1 extends BlackBox {
  override def desiredName = "xepic_ddr4_dc1"

  @public
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
    val s0_ddr4_s_axi          =
      new Ddr4AxiSlave(0)
    val s1_ddr4_s_axi          =
      new Ddr4AxiSlave(1)
  })

}

/**
 * Maps System DDR AXI onto the macro's 256-bit slave. Wide beats are split; 128-bit
 * transfers use narrow bursts, with lane (A + 16k)[4] selected for each beat.
 * CPU 0x80000000 is DDR offset 0, and the System's IDs widen to the macro's 11 bits.
 */
@instantiable
class P2EDdrAdapter(b: GlobalConfig) extends Module {
  val input  = b.device.ddr.port
  val system = input.copy(dataBits = input.dataBits.min(256))
  val base   = b.device.ddr.base

  require(system.dataBits == 128 || system.dataBits == 256)
  require(system.idBits <= XepicDdr4Dc1.portIdBits.head)

  @public
  val io = IO(new Bundle {
    val in  = Flipped(new axi4.Port(input))
    val out =
      Flipped(new Ddr4AxiSlave(0))
  })

  val bus = Wire(new axi4.Port(system))
  if (input.dataBits > system.dataBits) {
    val narrow = Instantiate(new axi4.Narrow(input, system))
    narrow.io.in <> io.in
    bus <> narrow.io.out
  } else bus <> io.in

  val lanes = 1 << system.idBits

  // Write lanes: W beats follow AW order, so queue each accepted burst's first lane.
  val writeLane = Instantiate(new Queue(Bool(), 8, false, false))

  val writeOdd   = RegInit(false.B)
  val writeUpper = writeLane.io.deq.bits ^ writeOdd
  io.out.awid            := bus.aw.bits.id
  io.out.awaddr          := bus.aw.bits.addr.pad(64) - base.U(64.W)
  io.out.awlen           := bus.aw.bits.len
  io.out.awsize          := bus.aw.bits.size
  io.out.awburst         := bus.aw.bits.burst
  io.out.awlock          := 0.U
  io.out.awcache         := "b0011".U
  io.out.awprot          := 0.U
  io.out.awqos           := 0.U
  io.out.awvalid         := bus.aw.valid && writeLane.io.enq.ready
  bus.aw.ready           := io.out.awready && writeLane.io.enq.ready
  writeLane.io.enq.valid := bus.aw.fire
  writeLane.io.enq.bits  := (if (system.dataBits == 128) bus.aw.bits.addr(4) else false.B)
  when(bus.aw.fire) {
    assert(
      bus.aw.bits.size === log2Ceil(system.bytes).U && bus.aw.bits.addr(
        log2Ceil(system.bytes) - 1,
        0
      ) === 0.U && bus.aw.bits.burst === 1.U,
      "P2E DDR adapter expects aligned whole-beat INCR bursts"
    )
  }

  io.out.wdata              := Fill(256 / system.dataBits, bus.w.bits.data)
  io.out.wstrb              := (if (system.dataBits == 128)
                     Mux(writeUpper, bus.w.bits.strb ## 0.U(16.W), 0.U(16.W) ## bus.w.bits.strb)
                   else bus.w.bits.strb)
  io.out.wlast              := bus.w.bits.last
  io.out.wvalid             := bus.w.valid && writeLane.io.deq.valid
  bus.w.ready               := io.out.wready && writeLane.io.deq.valid
  writeLane.io.deq.ready    := bus.w.fire && bus.w.bits.last
  when(bus.w.fire)(writeOdd := Mux(bus.w.bits.last, false.B, !writeOdd))

  bus.b.valid     := io.out.bvalid
  bus.b.bits.id   := io.out.bid(system.idBits - 1, 0)
  bus.b.bits.resp := io.out.bresp
  io.out.bready   := bus.b.ready

  // Read lanes: a System ID has at most one read in flight, so one lane bit per ID suffices.
  val readBusy  = RegInit(VecInit(Seq.fill(lanes)(false.B)))
  val readUpper = Reg(Vec(lanes, Bool()))
  io.out.arid    := bus.ar.bits.id
  io.out.araddr  := bus.ar.bits.addr.pad(64) - base.U(64.W)
  io.out.arlen   := bus.ar.bits.len
  io.out.arsize  := bus.ar.bits.size
  io.out.arburst := bus.ar.bits.burst
  io.out.arlock  := 0.U
  io.out.arcache := "b0011".U
  io.out.arprot  := 0.U
  io.out.arqos   := 0.U
  io.out.arvalid := bus.ar.valid
  bus.ar.ready   := io.out.arready
  when(bus.ar.fire) {
    assert(
      bus.ar.bits.size === log2Ceil(system.bytes).U && bus.ar.bits.addr(
        log2Ceil(system.bytes) - 1,
        0
      ) === 0.U && bus.ar.bits.burst === 1.U,
      "P2E DDR adapter expects aligned whole-beat INCR bursts"
    )
    assert(!readBusy(bus.ar.bits.id), "P2E DDR adapter allows one read in flight per ID")
    readBusy(bus.ar.bits.id)  := true.B
    readUpper(bus.ar.bits.id) := (if (system.dataBits == 128) bus.ar.bits.addr(4) else false.B)
  }

  val rid = io.out.rid(system.idBits - 1, 0)
  bus.r.valid     := io.out.rvalid
  bus.r.bits.id   := rid
  bus.r.bits.data := (if (system.dataBits == 128) Mux(readUpper(rid), io.out.rdata(255, 128), io.out.rdata(127, 0))
                      else io.out.rdata)
  bus.r.bits.resp := io.out.rresp
  bus.r.bits.last := io.out.rlast
  io.out.rready   := bus.r.ready
  when(bus.r.fire) {
    readUpper(rid)                      := !readUpper(rid)
    when(bus.r.bits.last)(readBusy(rid) := false.B)
  }
}

/**
 * The `top` instance bebop's flow addresses (`P2ETop.top.ddr`, `P2ETop.top.user_clk`): DDR4 macro,
 * DDR adapter and the System, all on `user_clk`. The SoC leaves reset only after calibration and
 * after the host releases `soc_hold`.
 */
@instantiable
class P2EHarness(b: GlobalConfig, target: SystemTarget) extends RawModule {
  val diffTest = b.sim.diffTest

  @public
  val io = FlatIO(new P2ETopIO)

  val ddr = Instantiate(new XepicDdr4Dc1)
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
    val soc     = Instantiate(new SimSoc(
      b,
      target
    ))
    val adapter = Instantiate(new P2EDdrAdapter(b))
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
class P2ETop(b: GlobalConfig, target: SystemTarget) extends RawModule {
  val diffTest = b.sim.diffTest

  val io = IO(new P2ETopIO)

  val top = Instantiate(new P2EHarness(
    b,
    target
  ))

  // The DDR4 data pins stay on the macro inside `top`: vcom binds the macro's own pins, and a top
  // inout driving the macro is rejected.
  for ((name, port) <- io.elements) port match {
    case _: Analog =>
    case _ => top.io.elements(name) <> port
  }
}
