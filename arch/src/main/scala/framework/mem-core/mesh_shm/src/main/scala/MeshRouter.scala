package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

object MeshDirection {
  val endpoint  = 0
  val east      = 1
  val west      = 2
  val south     = 3
  val north     = 4
  val portCount = 5
}

@instantiable
class MeshRouter(p: MeshSharedMemParams, row: Int, col: Int) extends Module {
  require(row >= 0 && row < p.rows && col >= 0 && col < p.cols)

  @public
  val io = IO(new Bundle {
    val in  = Vec(MeshDirection.portCount, Flipped(Decoupled(new MeshPacket(p))))
    val out = Vec(MeshDirection.portCount, Decoupled(new MeshPacket(p)))
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
          Mux(packet.destRow < row.U, MeshDirection.north.U, MeshDirection.endpoint.U)
        )
      )
    )
  }

  val choices = io.in.map(port => direction(port.bits))

  val arbiters = Seq.fill(MeshDirection.portCount)(Module(new RRArbiter(new MeshPacket(p), MeshDirection.portCount) {

    override lazy val lastGrant = {
      val pointer = RegInit(0.U(math.max(1, log2Ceil(MeshDirection.portCount)).W))
      when(io.out.fire)(pointer := io.chosen)
      pointer
    }

  }))

  for (output <- 0 until MeshDirection.portCount) {
    for (input <- 0 until MeshDirection.portCount) {
      arbiters(output).io.in(input).valid := io.in(input).valid && choices(input) === output.U
      arbiters(output).io.in(input).bits  := io.in(input).bits
    }
    val valid = RegInit(false.B)
    val data      = Reg(new MeshPacket(p))
    val heldValid = RegInit(false.B)
    val heldData  = Reg(new MeshPacket(p))
    val input     = arbiters(output).io.out
    val sink      = io.out(output)
    input.ready := !heldValid
    sink.valid  := valid
    sink.bits   := data
    when(!valid || sink.ready) {
      valid     := heldValid || input.fire
      when(heldValid || input.fire) {
        data := Mux(heldValid, heldData, input.bits)
      }
      heldValid := false.B
    }.elsewhen(input.fire) {
      heldValid := true.B
      heldData  := input.bits
    }
  }

  for (input <- 0 until MeshDirection.portCount) {
    io.in(input).ready := MuxLookup(choices(input), false.B)(
      (0 until MeshDirection.portCount).map(output => output.U -> arbiters(output).io.in(input).ready)
    )
    when(io.in(input).fire) {
      assert(io.in(input).bits.destRow < p.rows.U)
      assert(io.in(input).bits.destCol < p.cols.U)
    }
  }
}
