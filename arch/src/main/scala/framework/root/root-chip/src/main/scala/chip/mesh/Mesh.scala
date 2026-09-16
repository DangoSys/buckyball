package hier.chip.mesh

import chisel3._
import chisel3.util._

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

/**
 * Five-port deterministic XY router.
 *
 * Packet routes lock an output until tail. The router uses ready/valid inside
 * the IP; physical links may be wrapped by a credit adapter without changing
 * the routing contract.
 */
class MeshRouter(p: MeshParams, x: Int, y: Int) extends Module {
  require(x >= 0 && x < p.xNodes && y >= 0 && y < p.yNodes)
  private val flit      = new MeshFlit(p)
  private val inputBits = math.max(1, log2Ceil(MeshDirection.Count))

  val io = IO(new Bundle {
    val in  = Vec(MeshDirection.Count, Flipped(Decoupled(flit)))
    val out = Vec(MeshDirection.Count, Decoupled(flit))
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

  val lockValid = RegInit(VecInit(Seq.fill(MeshDirection.Count)(false.B)))
  val lockInput = Reg(Vec(MeshDirection.Count, UInt(inputBits.W)))
  val routes    = io.in.map(port => route(port.bits))

  for (out <- 0 until MeshDirection.Count) {
    val request  = VecInit((0 until MeshDirection.Count).map { in =>
      io.in(in).valid && routes(in) === out.U
    })
    val selected = Mux(lockValid(out), lockInput(out), PriorityEncoder(request))
    val valid    = Mux(lockValid(out), io.in(lockInput(out)).valid, request.asUInt.orR)
    io.out(out).valid := valid
    io.out(out).bits  := io.in(selected).bits

    when(io.out(out).fire) {
      when(lockValid(out) && io.out(out).bits.tail) {
        lockValid(out) := false.B
      }.elsewhen(!lockValid(out) && !io.out(out).bits.tail) {
        lockValid(out) := true.B
        lockInput(out) := selected
      }
    }
  }

  for (in <- 0 until MeshDirection.Count) {
    val selectedOut = routes(in)
    io.in(in).ready := io.out(selectedOut).ready && io.out(selectedOut).valid &&
      Mux(lockValid(selectedOut), lockInput(selectedOut) === in.U, true.B)
    when(io.in(in).fire && !io.in(in).bits.head) {
      assert(
        lockValid(selectedOut) && lockInput(selectedOut) === in.U,
        "Mesh packet body arrived without the output lock"
      )
    }
  }
}

/** Local endpoint arbiter. Once a packet starts, its source retains the mesh port to tail. */
class MeshPacketArbiter(p: MeshParams, inputs: Int) extends Module {
  require(inputs >= 1)
  private val choiceBits = math.max(1, log2Ceil(inputs))

  val io = IO(new Bundle {
    val in  = Vec(inputs, Flipped(Decoupled(new MeshFlit(p))))
    val out = Decoupled(new MeshFlit(p))
  })

  val locked      = RegInit(false.B)
  val selected    = RegInit(0.U(choiceBits.W))
  val validInputs = VecInit(io.in.map(_.valid))
  val choice      = PriorityEncoder(validInputs)
  val active      = Mux(locked, selected, choice)
  io.out.valid := Mux(locked, io.in(selected).valid, validInputs.asUInt.orR)
  io.out.bits  := io.in(active).bits
  for (index <- 0 until inputs) {
    io.in(index).ready := io.out.ready && io.out.valid && active === index.U
  }
  when(io.out.fire) {
    when(locked && io.out.bits.tail) {
      locked := false.B
    }.elsewhen(!locked && !io.out.bits.tail) {
      locked   := true.B
      selected := active
    }
  }
}

/** 2D mesh of independent routers with buffered directed links. */
class MeshNetwork(p: MeshParams, linkDepth: Int = 2) extends Module {
  require(linkDepth >= 1)
  private val flit = new MeshFlit(p)

  private val routers = Seq.tabulate(p.yNodes, p.xNodes) { case (y, x) =>
    Module(new MeshRouter(p, x, y))
  }

  private def index(x: Int, y: Int): Int = y * p.xNodes + x

  val io = IO(new Bundle {
    val localIn  = Vec(p.xNodes * p.yNodes, Flipped(Decoupled(flit)))
    val localOut = Vec(p.xNodes * p.yNodes, Decoupled(flit))
  })

  def connect(source: DecoupledIO[MeshFlit], sink: DecoupledIO[MeshFlit]): Unit = {
    sink.valid   := source.valid
    sink.bits    := source.bits
    source.ready := sink.ready
  }

  for {
    y <- 0 until p.yNodes
    x <- 0 until p.xNodes
  } {
    val router = routers(y)(x)
    val local  = index(x, y)
    connect(io.localIn(local), router.io.in(MeshDirection.Local))
    connect(router.io.out(MeshDirection.Local), io.localOut(local))

    if (x == 0) {
      router.io.in(MeshDirection.West).valid  := false.B
      router.io.in(MeshDirection.West).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.West).ready := true.B
    } else {
      val link = Module(new Queue(flit, linkDepth))
      connect(routers(y)(x - 1).io.out(MeshDirection.East), link.io.enq)
      connect(link.io.deq, router.io.in(MeshDirection.West))
    }

    if (x == p.xNodes - 1) {
      router.io.in(MeshDirection.East).valid  := false.B
      router.io.in(MeshDirection.East).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.East).ready := true.B
    } else {
      val link = Module(new Queue(flit, linkDepth))
      connect(routers(y)(x + 1).io.out(MeshDirection.West), link.io.enq)
      connect(link.io.deq, router.io.in(MeshDirection.East))
    }

    if (y == 0) {
      router.io.in(MeshDirection.North).valid  := false.B
      router.io.in(MeshDirection.North).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.North).ready := true.B
    } else {
      val link = Module(new Queue(flit, linkDepth))
      connect(routers(y - 1)(x).io.out(MeshDirection.South), link.io.enq)
      connect(link.io.deq, router.io.in(MeshDirection.North))
    }

    if (y == p.yNodes - 1) {
      router.io.in(MeshDirection.South).valid  := false.B
      router.io.in(MeshDirection.South).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.South).ready := true.B
    } else {
      val link = Module(new Queue(flit, linkDepth))
      connect(routers(y + 1)(x).io.out(MeshDirection.North), link.io.enq)
      connect(link.io.deq, router.io.in(MeshDirection.South))
    }
  }
}
