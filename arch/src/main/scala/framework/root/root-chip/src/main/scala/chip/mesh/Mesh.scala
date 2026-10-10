package hier.chip.mesh

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

object MeshDirection {
  val Local = 0
  val North = 1
  val South = 2
  val East  = 3
  val West  = 4
  val Count = 5
}

case class MeshParams(
  xNodes:          Int,
  yNodes:          Int,
  payloadBits:     Int,
  virtualChannels: Int = 2) {
  require(xNodes >= 1 && yNodes >= 1)
  require(payloadBits > 0)
  require(virtualChannels >= 1 && isPow2(virtualChannels))
  val xBits  = math.max(1, log2Ceil(xNodes))
  val yBits  = math.max(1, log2Ceil(yNodes))
  val vcBits = math.max(1, log2Ceil(virtualChannels))
}

/** Generic wormhole flit. Packet content is opaque to the mesh. */
class MeshFlit(p: MeshParams) extends Bundle {
  val srcX    = UInt(p.xBits.W)
  val srcY    = UInt(p.yBits.W)
  val dstX    = UInt(p.xBits.W)
  val dstY    = UInt(p.yBits.W)
  val vc      = UInt(p.vcBits.W)
  val head    = Bool()
  val tail    = Bool()
  val payload = UInt(p.payloadBits.W)
}

/** Deterministic XY routing; only physically connected directions own state. */
@instantiable
class MeshRouter(p: MeshParams, x: Int, y: Int) extends Module {
  require(x >= 0 && x < p.xNodes && y >= 0 && y < p.yNodes)
  val flit = new MeshFlit(p)

  val connected = Seq(MeshDirection.Local) ++
    Seq(MeshDirection.North).filter(_ => y > 0) ++
    Seq(MeshDirection.South).filter(_ => y < p.yNodes - 1) ++
    Seq(MeshDirection.East).filter(_ => x < p.xNodes - 1) ++
    Seq(MeshDirection.West).filter(_ => x > 0)

  val count     = connected.size
  val inputBits = math.max(1, log2Ceil(count))

  @public
  val io = IO(new Bundle {
    val in  = Vec(MeshDirection.Count, Vec(p.virtualChannels, Flipped(Decoupled(flit))))
    val out = Vec(MeshDirection.Count, Vec(p.virtualChannels, Decoupled(flit)))
  })

  def route(value: MeshFlit): UInt = Mux(
    value.dstX < x.U,
    MeshDirection.West.U,
    Mux(
      value.dstX > x.U,
      MeshDirection.East.U,
      Mux(value.dstY < y.U, MeshDirection.North.U, Mux(value.dstY > y.U, MeshDirection.South.U, MeshDirection.Local.U))
    )
  )

  for {
    direction <- (0 until MeshDirection.Count).filterNot(connected.contains)
    vc        <- 0 until p.virtualChannels
  } {
    io.in(direction)(vc).ready  := false.B
    io.out(direction)(vc).valid := false.B
    io.out(direction)(vc).bits  := 0.U.asTypeOf(flit)
  }
  for (vc     <- 0 until p.virtualChannels) {
    val inputs   = connected.map(direction => io.in(direction)(vc))
    val payloads = VecInit(inputs.map(_.bits))
    val routes   = inputs.map(input => route(input.bits))
    val owners   = connected.map { out =>
      val lockValid  = RegInit(false.B)
      val lockInput  = Reg(UInt(inputBits.W))
      val cursor     = RegInit(0.U(inputBits.W))
      val request    = VecInit(inputs.indices.map(in => inputs(in).valid && routes(in) === out.U))
      val ordered    = VecInit(inputs.indices.map(offset => request((cursor +& offset.U) % count.U)))
      val fairChoice = (cursor +& PriorityEncoder(ordered)) % count.U
      val selected   = Mux(lockValid, lockInput, fairChoice)
      val valid      = Mux(lockValid, VecInit(inputs.map(_.valid))(lockInput), request.asUInt.orR)
      io.out(out)(vc).valid := valid
      io.out(out)(vc).bits  := payloads(selected)
      when(valid && !io.out(out)(vc).ready && !lockValid) {
        lockValid := true.B
        lockInput := selected
      }
      when(io.out(out)(vc).fire) {
        when(io.out(out)(vc).bits.tail) {
          lockValid := false.B
          cursor    := Mux(selected === (count - 1).U, 0.U, selected + 1.U)
        }.otherwise {
          lockValid := true.B
          lockInput := selected
        }
      }
      (out, selected, lockValid, lockInput)
    }
    for ((input, index) <- inputs.zipWithIndex) {
      input.ready := owners.map { case (out, selected, _, _) =>
        io.out(out)(vc).ready && io.out(out)(vc).valid && selected === index.U
      }.reduce(_ || _)
      when(input.fire) {
        assert(input.bits.vc === vc.U, "Mesh flit arrived on wrong VC port")
        assert(input.bits.dstX < p.xNodes.U && input.bits.dstY < p.yNodes.U, "Mesh destination outside topology")
        when(!input.bits.head) {
          val owned = owners.map { case (out, _, locked, owner) =>
            routes(index) === out.U && locked && owner === index.U
          }.reduce(_ || _)
          assert(owned, "Mesh packet body without ownership")
        }
      }
    }
  }
}
