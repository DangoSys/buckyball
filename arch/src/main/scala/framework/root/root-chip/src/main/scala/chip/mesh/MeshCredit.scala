package hier.chip.mesh

import chisel3._
import chisel3.util._

/** One directed physical Mesh link. Each VC returns its own buffer credits. */
class MeshCreditLink(p: MeshParams) extends Bundle {
  val valid  = Output(Bool())
  val flit   = Output(new MeshFlit(p))
  val credit = Input(Vec(p.virtualChannels, Bool()))
}

/**
 * Sender half of a credit-based Mesh link.
 *
 * Credits start at zero and are supplied by the receiver after link activation.
 * A coordinated reset is the only supported way to stop an active link.
 */
class MeshCreditTx(p: MeshParams, maxCredits: Int) extends Module {
  require(maxCredits >= 1 && maxCredits <= 255)
  private val creditBits = math.max(1, log2Ceil(maxCredits + 1))

  val io = IO(new Bundle {
    val active  = Input(Bool())
    val in      = Flipped(Decoupled(new MeshFlit(p)))
    val link    = new MeshCreditLink(p)
    val credits = Output(UInt(creditBits.W))
  })

  val credits  = RegInit(VecInit(Seq.fill(p.virtualChannels)(0.U(creditBits.W))))
  val returned = RegNext(io.link.credit, VecInit(Seq.fill(p.virtualChannels)(false.B)))
  io.in.ready := io.active && credits(io.in.bits.vc) =/= 0.U
  val send = io.in.fire
  for (vc <- 0 until p.virtualChannels) {
    val consumed = send && io.in.bits.vc === vc.U
    when(returned(vc) =/= consumed) {
      credits(vc) := Mux(returned(vc), credits(vc) + 1.U, credits(vc) - 1.U)
    }
    when(returned(vc) && !consumed)(assert(credits(vc) < maxCredits.U, "Mesh TX credit overflow"))
    when(consumed)(assert(credits(vc) =/= 0.U, "Mesh TX credit underflow"))
  }
  io.link.valid := RegNext(send, false.B)
  io.link.flit := RegEnable(io.in.bits, 0.U.asTypeOf(new MeshFlit(p)), send)
  io.credits   := credits(io.in.bits.vc)
  val wasActive = RegNext(io.active, false.B)
  when(wasActive)(assert(io.active, "Mesh TX requires coordinated reset to stop"))
}

/** Receiver half of a credit-based Mesh link with independent FIFO and credit pool per VC. */
class MeshCreditRx(p: MeshParams, depth: Int) extends Module {
  require(depth >= 1 && depth <= 255)
  private val creditBits = math.max(1, log2Ceil(depth + 1))

  val io = IO(new Bundle {
    val active    = Input(Bool())
    val link      = Flipped(new MeshCreditLink(p))
    val out       = Decoupled(new MeshFlit(p))
    val occupancy = Output(Vec(p.virtualChannels, UInt(creditBits.W)))
  })

  val queues       = Seq.fill(p.virtualChannels)(Module(new Queue(new MeshFlit(p), depth, pipe = true)))
  val advertised   = RegInit(VecInit(Seq.fill(p.virtualChannels)(0.U(creditBits.W))))
  val arbiter      = Module(new RRArbiter(new MeshFlit(p), p.virtualChannels))
  val enqueueReady = Wire(Vec(p.virtualChannels, Bool()))
  io.out <> arbiter.io.out
  for (vc <- 0 until p.virtualChannels) {
    val queue    = queues(vc)
    val accepted = io.link.valid && io.link.flit.vc === vc.U
    val grant    = io.active && (advertised(vc) +& queue.io.count) < depth.U
    io.link.credit(vc) := grant
    queue.io.enq.valid := accepted
    queue.io.enq.bits  := io.link.flit
    enqueueReady(vc)   := queue.io.enq.ready
    arbiter.io.in(vc) <> queue.io.deq
    when(grant =/= accepted) {
      advertised(vc) := Mux(grant, advertised(vc) + 1.U, advertised(vc) - 1.U)
    }
    io.occupancy(vc)   := queue.io.count
  }
  when(io.link.valid) {
    assert(io.active, "Mesh RX flit on inactive link")
    assert(advertised(io.link.flit.vc) =/= 0.U, "Mesh RX flit without returned credit")
    assert(enqueueReady(io.link.flit.vc), "Mesh RX buffer overflow")
  }
  for (vc <- 0 until p.virtualChannels) {
    assert((advertised(vc) +& queues(vc).io.count) <= depth.U, "Mesh RX credit conservation")
  }
  val wasActive = RegNext(io.active, false.B)
  when(wasActive)(assert(io.active, "Mesh RX requires coordinated reset to stop"))
}

/** Native verification target for one credit link. */
class MeshCreditLoopback extends Module {
  private val mesh = MeshParams(xNodes = 2, yNodes = 2, payloadBits = 32, virtualChannels = 8)

  val io = IO(new Bundle {
    val active    = Input(Bool())
    val in        = Flipped(Decoupled(new MeshFlit(mesh)))
    val out       = Decoupled(new MeshFlit(mesh))
    val credits   = Output(UInt(3.W))
    val occupancy = Output(UInt(3.W))
  })

  val tx = Module(new MeshCreditTx(mesh, maxCredits = 4))
  val rx = Module(new MeshCreditRx(mesh, depth = 4))
  tx.io.active := io.active
  rx.io.active := io.active
  tx.io.in <> io.in
  rx.io.link <> tx.io.link
  io.out <> rx.io.out
  io.credits   := tx.io.credits
  io.occupancy := rx.io.occupancy(3)
}

/** 2D Mesh whose inter-router links use per-VC credit flow control. */
class MeshCreditNetwork(p: MeshParams, linkDepth: Int = 2) extends Module {
  require(linkDepth >= 1)
  private val flit = new MeshFlit(p)

  private val routers = Seq.tabulate(p.yNodes, p.xNodes) { case (y, x) =>
    Module(new MeshRouter(p, x, y))
  }

  private def index(x: Int, y: Int): Int = y * p.xNodes + x

  val io = IO(new Bundle {
    val active   = Input(Bool())
    val localIn  = Vec(p.xNodes * p.yNodes, Flipped(Decoupled(flit)))
    val localOut = Vec(p.xNodes * p.yNodes, Decoupled(flit))
  })

  def wireLink(source: DecoupledIO[MeshFlit], sink: DecoupledIO[MeshFlit]): Unit = {
    val tx = Module(new MeshCreditTx(p, linkDepth))
    val rx = Module(new MeshCreditRx(p, linkDepth))
    tx.io.active := io.active
    rx.io.active := io.active
    tx.io.in <> source
    rx.io.link <> tx.io.link
    sink <> rx.io.out
  }

  for {
    y <- 0 until p.yNodes
    x <- 0 until p.xNodes
  } {
    val router = routers(y)(x)
    router.io.in(MeshDirection.Local) <> io.localIn(index(x, y))
    io.localOut(index(x, y)) <> router.io.out(MeshDirection.Local)
    if (x == 0) {
      router.io.in(MeshDirection.West).valid  := false.B
      router.io.in(MeshDirection.West).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.West).ready := true.B
    }
    if (x == p.xNodes - 1) {
      router.io.in(MeshDirection.East).valid  := false.B
      router.io.in(MeshDirection.East).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.East).ready := true.B
    }
    if (y == 0) {
      router.io.in(MeshDirection.North).valid  := false.B
      router.io.in(MeshDirection.North).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.North).ready := true.B
    }
    if (y == p.yNodes - 1) {
      router.io.in(MeshDirection.South).valid  := false.B
      router.io.in(MeshDirection.South).bits   := 0.U.asTypeOf(flit)
      router.io.out(MeshDirection.South).ready := true.B
    }
  }
  for {
    y <- 0 until p.yNodes
    x <- 0 until p.xNodes - 1
  } {
    wireLink(routers(y)(x).io.out(MeshDirection.East), routers(y)(x + 1).io.in(MeshDirection.West))
    wireLink(routers(y)(x + 1).io.out(MeshDirection.West), routers(y)(x).io.in(MeshDirection.East))
  }
  for {
    y <- 0 until p.yNodes - 1
    x <- 0 until p.xNodes
  } {
    wireLink(routers(y)(x).io.out(MeshDirection.South), routers(y + 1)(x).io.in(MeshDirection.North))
    wireLink(routers(y + 1)(x).io.out(MeshDirection.North), routers(y)(x).io.in(MeshDirection.South))
  }
}
