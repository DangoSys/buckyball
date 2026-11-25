package framework.device.peripherals.clint

import chisel3._
import framework.device.peripherals.clint.configs.ClintParams
import framework.device.peripherals.DevicePort
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import chisel3.util._

/** `mtime` advances once every `tickCycles` clock cycles; the device tree timebase must match. */
/** Standard CLINT layout: msip at 4*hart, mtimecmp at 0x4000 + 8*hart, mtime at 0xbff8. */
@instantiable
class Clint(p: ClintParams) extends Module {
  val hartIds = p.hartIds
  require(hartIds.distinct.size == hartIds.size && hartIds.max < 4095)
  val n       = hartIds.size

  @public
  val io = IO(new Bundle {
    val port = new DevicePort
    val msip = Output(Vec(n, Bool()))
    val mtip = Output(Vec(n, Bool()))
  })

  val msip     = RegInit(VecInit(Seq.fill(n)(false.B)))
  val mtimecmp = RegInit(VecInit(Seq.fill(n)(~0.U(64.W))))
  val mtime    = RegInit(0.U(64.W))
  val ticks    = RegInit(0.U(log2Ceil(p.tickCycles + 1).W))

  val access = io.port.access.bits
  val offset = access.addr - p.base.U
  val write  = io.port.access.valid && access.write
  val word   = access.size === 2.U
  val double = access.size === 3.U

  // Each register answers a 32-bit access to either half, and a 64-bit access to its base.
  val hits  = Wire(Vec(2 * n + 1, Bool()))
  val reads = Wire(Vec(2 * n + 1, UInt(64.W)))

  def register(
    slot:   Int,
    at:     BigInt,
    state:  UInt,
    update: UInt => Unit,
    wide:   Boolean
  ): Unit = {
    val value = WireDefault(UInt(64.W), state)
    val low   = word && offset === at.U
    val high  = wide.B && word && offset === (at + 4).U
    val full  = wide.B && double && offset === at.U
    hits(slot)  := low || high || full
    reads(slot) := Mux(high, value(63, 32), Mux(low, value(31, 0), value))
    when(write && low)(update(if (wide) Cat(value(63, 32), access.data(31, 0)) else access.data(31, 0)))
    when(write && high)(update(Cat(access.data(31, 0), value(31, 0))))
    when(write && full)(update(access.data))
  }

  ticks                                      := Mux(ticks === (p.tickCycles - 1).U, 0.U, ticks + 1.U)
  when(ticks === (p.tickCycles - 1).U)(mtime := mtime + 1.U)
  for ((hart, i) <- hartIds.zipWithIndex) {
    register(i, 4 * hart, msip(i).asUInt, v => msip(i) := v(0), wide = false)
    register(n + i, 0x4000 + 8 * BigInt(hart), mtimecmp(i), v => mtimecmp(i) := v, wide = true)
    io.msip(i) := msip(i)
    io.mtip(i) := mtime >= mtimecmp(i)
  }
  register(2 * n, 0xbff8, mtime, v => mtime := v, wide = true)

  io.port.read := Mux1H(hits, reads)
  io.port.error := !hits.asUInt.orR
}
