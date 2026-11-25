package framework.generic.acc.memdomain

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.memory.interlock.{Acknowledgement, Maintenance, MaintenanceOp, Params}

/**
 * Sends page-bounded ranges to a capable Home; otherwise walks and filters individual lines.
 */
@instantiable
class RangeWalker(p: Params, homeRanges: Boolean) extends Module {

  @public
  val io = IO(new Bundle {
    val range        = Flipped(Decoupled(new Maintenance(p)))
    val done         = Decoupled(new Acknowledgement(p))
    val maintenance  = Decoupled(new Maintenance(p))
    val maintained   = Flipped(Decoupled(new Acknowledgement(p)))
    val lineAddress  = Output(UInt(p.addressBits.W))
    val lineWrite    = Output(Bool())
    val lineMaintain = Input(Bool())
    val drained      = Input(Bool())
  })

  val idle :: scan :: acknowledge :: finish :: Nil = Enum(4)
  val state                                        = RegInit(idle)
  val range                                        = Reg(new Maintenance(p))
  val cursor                                       = Reg(UInt(p.addressBits.W))
  val succeeded                                    = Reg(Bool())

  io.range.ready := state === idle && io.drained
  io.lineAddress := cursor
  io.lineWrite   := range.op =/= MaintenanceOp.Clean.U
  when(io.range.fire) {
    range     := io.range.bits
    cursor    := io.range.bits.firstLine
    succeeded := true.B
    state     := scan
  }

  val last =
    if (homeRanges) {
      val pageLast = Cat(cursor(p.addressBits - 1, 12), "hfc0".U(12.W))
      Mux(range.lastLine < pageLast, range.lastLine, pageLast)
    } else cursor

  val maintain = if (homeRanges) true.B else io.lineMaintain

  io.maintenance.valid               := state === scan && maintain
  io.maintenance.bits                := range
  io.maintenance.bits.firstLine      := cursor
  io.maintenance.bits.lastLine       := last
  when(io.maintenance.fire)(state    := acknowledge)
  io.maintained.ready                := state === acknowledge
  when(io.maintained.fire)(succeeded := succeeded && io.maintained.bits.ok)

  when(state === scan && !maintain || io.maintained.fire) {
    cursor := last + p.lineBytes.U
    state  := Mux(last === range.lastLine, finish, scan)
  }

  io.done.valid            := state === finish
  io.done.bits.tag         := range.tag
  io.done.bits.ok          := succeeded
  when(io.done.fire)(state := idle)
}
