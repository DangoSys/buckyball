package framework.ip.bus.axi4

import framework.ip.bus.axi4.{Params => AxiParam}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** Address-selected AXI branch. One read burst and one write burst may be live independently. */
@instantiable
class Decoder(val regions: Seq[(BigInt, BigInt)], axiParam: AxiParam) extends Module {
  val p = axiParam

  @public
  val io = IO(new Bundle {
    val in      = Flipped(new Port(axiParam))
    val matched = new Port(axiParam)
    val other   = new Port(axiParam)
  })

  val readBusy         = RegInit(false.B)
  val writeDataPending = RegInit(false.B)
  val writeBusy        = RegInit(false.B)
  val readMatched      = Reg(Bool())
  val writeMatched     = Reg(Bool())

  val arMatched = regions.map { case (base, bytes) =>
    io.in.ar.bits.addr >= base.U && io.in.ar.bits.addr < (base + bytes).U
  }.reduce(_ || _)

  val awMatched = regions.map { case (base, bytes) =>
    io.in.aw.bits.addr >= base.U && io.in.aw.bits.addr < (base + bytes).U
  }.reduce(_ || _)

  io.matched.ar.valid                              := io.in.ar.valid && !readBusy && arMatched
  io.other.ar.valid                                := io.in.ar.valid && !readBusy && !arMatched
  io.matched.ar.bits                               := io.in.ar.bits
  io.other.ar.bits                                 := io.in.ar.bits
  io.in.ar.ready                                   := !readBusy && Mux(arMatched, io.matched.ar.ready, io.other.ar.ready)
  when(io.in.ar.fire) { readBusy := true.B; readMatched := arMatched }
  io.in.r.valid                                    := readBusy && Mux(readMatched, io.matched.r.valid, io.other.r.valid)
  io.in.r.bits                                     := Mux(readMatched, io.matched.r.bits, io.other.r.bits)
  io.matched.r.ready                               := readBusy && readMatched && io.in.r.ready
  io.other.r.ready                                 := readBusy && !readMatched && io.in.r.ready
  when(io.in.r.fire && io.in.r.bits.last)(readBusy := false.B)

  io.matched.aw.valid                                      := io.in.aw.valid && !writeBusy && awMatched
  io.other.aw.valid                                        := io.in.aw.valid && !writeBusy && !awMatched
  io.matched.aw.bits                                       := io.in.aw.bits
  io.other.aw.bits                                         := io.in.aw.bits
  io.in.aw.ready                                           := !writeBusy && Mux(awMatched, io.matched.aw.ready, io.other.aw.ready)
  when(io.in.aw.fire) { writeBusy := true.B; writeDataPending := true.B; writeMatched := awMatched }
  io.matched.w.valid                                       := io.in.w.valid && writeBusy && writeDataPending && writeMatched
  io.other.w.valid                                         := io.in.w.valid && writeBusy && writeDataPending && !writeMatched
  io.matched.w.bits                                        := io.in.w.bits
  io.other.w.bits                                          := io.in.w.bits
  io.in.w.ready                                            := writeBusy && writeDataPending && Mux(writeMatched, io.matched.w.ready, io.other.w.ready)
  when(io.in.w.fire && io.in.w.bits.last)(writeDataPending := false.B)
  io.in.b.valid                                            := writeBusy && Mux(writeMatched, io.matched.b.valid, io.other.b.valid)
  io.in.b.bits                                             := Mux(writeMatched, io.matched.b.bits, io.other.b.bits)
  io.matched.b.ready                                       := writeBusy && writeMatched && io.in.b.ready
  io.other.b.ready                                         := writeBusy && !writeMatched && io.in.b.ready
  when(io.in.b.fire)(writeBusy                             := false.B)
}
