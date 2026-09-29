package examples.poly.meshsharedmem

import chisel3._
import chisel3.util._

object MeshDirection {
  val local = 0
  val east  = 1
  val west  = 2
  val south = 3
  val north = 4
  val count = 5
}

class MeshRouter(p: MeshSharedMemParams, row: Int, col: Int) extends Module {
  require(row >= 0 && row < p.rows && col >= 0 && col < p.cols)

  val io = IO(new Bundle {
    val in  = Vec(MeshDirection.count, Flipped(Decoupled(new MeshPacket(p))))
    val out = Vec(MeshDirection.count, Decoupled(new MeshPacket(p)))
  })

  def direction(packet: MeshPacket): UInt = {
    Mux(
      packet.destCol > col.U,
      MeshDirection.east.U,
      Mux(
        packet.destCol < col.U,
        MeshDirection.west.U,
        Mux(
          packet.destRow > row.U,
          MeshDirection.south.U,
          Mux(packet.destRow < row.U, MeshDirection.north.U, MeshDirection.local.U)
        )
      )
    )
  }

  val choices  = io.in.map(port => direction(port.bits))
  val arbiters = Seq.fill(MeshDirection.count)(Module(new RRArbiter(new MeshPacket(p), MeshDirection.count)))

  for (output <- 0 until MeshDirection.count) {
    for (input <- 0 until MeshDirection.count) {
      arbiters(output).io.in(input).valid := io.in(input).valid && choices(input) === output.U
      arbiters(output).io.in(input).bits  := io.in(input).bits
    }
    io.out(output) <> arbiters(output).io.out
  }

  for (input <- 0 until MeshDirection.count) {
    io.in(input).ready := MuxLookup(choices(input), false.B)(
      (0 until MeshDirection.count).map(output => output.U -> arbiters(output).io.in(input).ready)
    )
    when(io.in(input).fire) {
      assert(io.in(input).bits.destRow < p.rows.U)
      assert(io.in(input).bits.destCol < p.cols.U)
    }
  }
}
