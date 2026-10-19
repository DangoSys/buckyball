package memcore.memory.mesh_shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}

object MeshDirection {
  val endpoint  = 0
  val east      = 1
  val west      = 2
  val south     = 3
  val north     = 4
  val portCount = 5
}

@instantiable
class MeshCreditRouter(p: MeshSharedMemParams, row: Int, col: Int) extends Module {
  require(row >= 0 && row < p.rows && col >= 0 && col < p.cols)

  @public
  val io = IO(new Bundle {
    val in  = Vec(MeshDirection.portCount, Flipped(new MeshCreditPort(p)))
    val out = Vec(MeshDirection.portCount, new MeshCreditPort(p))
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

  val inputs = Seq.fill(MeshDirection.portCount) {
    Module(new Queue(new MeshPacket(p), p.linkBufferDepth, pipe = false, flow = false))
  }

  for ((input, port) <- inputs.zip(io.in)) {
    input.io.enq.valid := port.valid
    input.io.enq.bits  := port.bits
    port.credit        := input.io.deq.fire
    when(port.valid) {
      assert(input.io.enq.ready, "Mesh credit sender exceeded the router input capacity")
    }
  }

  val choices = inputs.map(input => direction(input.io.deq.bits))

  val arbiters = Seq.fill(MeshDirection.portCount)(Module(new RRArbiter(new MeshPacket(p), MeshDirection.portCount) {

    override lazy val lastGrant = {
      val pointer = RegInit(0.U(math.max(1, log2Ceil(MeshDirection.portCount)).W))
      when(io.out.fire)(pointer := io.chosen)
      pointer
    }

  }))

  for (output <- 0 until MeshDirection.portCount) {
    for (input <- 0 until MeshDirection.portCount) {
      arbiters(output).io.in(input).valid := inputs(input).io.deq.valid && choices(input) === output.U
      arbiters(output).io.in(input).bits  := inputs(input).io.deq.bits
    }
    val credits = RegInit(p.linkBufferDepth.U(p.linkCreditBits.W))
    val available = credits =/= 0.U
    val sent      = arbiters(output).io.out.valid && available
    io.out(output).valid          := sent
    io.out(output).bits           := arbiters(output).io.out.bits
    arbiters(output).io.out.ready := available
    when(io.out(output).credit && !sent) {
      assert(credits < p.linkBufferDepth.U)
      credits := credits + 1.U
    }.elsewhen(sent && !io.out(output).credit) {
      credits := credits - 1.U
    }
  }

  for (input <- 0 until MeshDirection.portCount) {
    inputs(input).io.deq.ready := MuxLookup(choices(input), false.B)(
      (0 until MeshDirection.portCount).map(output => output.U -> arbiters(output).io.in(input).ready)
    )
    when(inputs(input).io.deq.fire) {
      assert(inputs(input).io.deq.bits.destRow < p.rows.U)
      assert(inputs(input).io.deq.bits.destCol < p.cols.U)
    }
  }
}

@instantiable
class MeshRouter(p: MeshSharedMemParams, row: Int, col: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val in  = Vec(MeshDirection.portCount, Flipped(Decoupled(new MeshPacket(p))))
    val out = Vec(MeshDirection.portCount, Decoupled(new MeshPacket(p)))
  })

  val router = Instantiate(new MeshCreditRouter(p, row, col))
  for (port <- 0 until MeshDirection.portCount) {
    val source = Instantiate(new MeshDecoupledToCredit(p))
    val sink   = Instantiate(new MeshCreditToDecoupled(p))
    source.io.in <> io.in(port)
    router.io.in(port) <> source.io.out
    sink.io.in <> router.io.out(port)
    io.out(port) <> sink.io.out
  }
}
