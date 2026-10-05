package hier.chip.mesh

import memcore.memory.queue.Queue

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

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
@instantiable
class MeshCreditTx(p: MeshParams, maxCredits: Int) extends Module {
  require(maxCredits >= 1 && maxCredits <= 255)
  private val creditBits = math.max(1, log2Ceil(maxCredits + 1))

  @public
  val io = IO(new Bundle {
    val active  = Input(Bool())
    val in      = Vec(p.virtualChannels, Flipped(Decoupled(new MeshFlit(p))))
    val link    = new MeshCreditLink(p)
    val credits = Output(UInt(creditBits.W))
  })

  val credits  = RegInit(VecInit(Seq.fill(p.virtualChannels)(0.U(creditBits.W))))
  val returned = RegNext(io.link.credit, VecInit(Seq.fill(p.virtualChannels)(false.B)))
  val cursor   = RegInit(0.U(p.vcBits.W))
  val eligible = VecInit((0 until p.virtualChannels).map(vc => io.active && io.in(vc).valid && credits(vc) =/= 0.U))
  val ordered  = VecInit((0 until p.virtualChannels).map(offset => eligible((cursor +& offset.U) % p.virtualChannels.U)))
  val selected = (cursor +& PriorityEncoder(ordered)) % p.virtualChannels.U
  val send     = eligible.asUInt.orR
  for (vc <- 0 until p.virtualChannels) {
    io.in(vc).ready := send && selected === vc.U
    val consumed = send && selected === vc.U
    when(returned(vc) =/= consumed)(credits(vc) := Mux(returned(vc), credits(vc) + 1.U, credits(vc) - 1.U))
    when(returned(vc) && !consumed)(assert(credits(vc) < maxCredits.U, "Mesh TX credit overflow"))
    when(consumed)(assert(io.in(vc).bits.vc === vc.U, "Mesh TX wrong VC port"))
  }
  when(send)(cursor := Mux(selected === (p.virtualChannels - 1).U, 0.U, selected + 1.U))
  io.link.valid := RegNext(send, false.B)
  io.link.flit  := RegEnable(io.in(selected).bits, 0.U.asTypeOf(new MeshFlit(p)), send)
  io.credits    := credits(selected)
  val wasActive = RegNext(io.active, false.B)
  when(wasActive)(assert(io.active, "Mesh TX requires coordinated reset to stop"))
}

/** Receiver half of a credit-based Mesh link with independent FIFO and credit pool per VC. */
@instantiable
class MeshCreditRx(p: MeshParams, depth: Int) extends Module {
  require(depth >= 1 && depth <= 255)
  private val creditBits = math.max(1, log2Ceil(depth + 1))

  @public
  val io = IO(new Bundle {
    val active    = Input(Bool())
    val link      = Flipped(new MeshCreditLink(p))
    val out       = Vec(p.virtualChannels, Decoupled(new MeshFlit(p)))
    val occupancy = Output(Vec(p.virtualChannels, UInt(creditBits.W)))
  })

  val queues       = Seq.fill(p.virtualChannels)(Module(new Queue(new MeshFlit(p), depth, pipe = true)))
  val advertised   = RegInit(VecInit(Seq.fill(p.virtualChannels)(0.U(creditBits.W))))
  val enqueueReady = Wire(Vec(p.virtualChannels, Bool()))
  for (vc <- 0 until p.virtualChannels) {
    val queue    = queues(vc)
    val accepted = io.link.valid && io.link.flit.vc === vc.U
    val grant    = io.active && (advertised(vc) +& queue.io.count) < depth.U
    io.link.credit(vc) := grant
    queue.io.enq.valid := accepted
    queue.io.enq.bits  := io.link.flit
    enqueueReady(vc)   := queue.io.enq.ready
    io.out(vc) <> queue.io.deq
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

/** 2D Mesh with explicit independent local VC ports and per-VC physical credits. */
@instantiable
class MeshCreditNetwork(p: MeshParams, linkDepth: Int = 2) extends Module {
  require(linkDepth >= 1 && linkDepth <= 255)
  private val flit = new MeshFlit(p)
  private val routers: Seq[Seq[Instance[MeshRouter]]] =
    Seq.tabulate(p.yNodes, p.xNodes)((y, x) => Instantiate(new MeshRouter(p, x, y)))
  private def index(x: Int, y: Int): Int = y * p.xNodes + x

  @public
  val io = IO(new Bundle {
    val active   = Input(Bool())
    val localIn  = Vec(p.xNodes * p.yNodes, Vec(p.virtualChannels, Flipped(Decoupled(flit))))
    val localOut = Vec(p.xNodes * p.yNodes, Vec(p.virtualChannels, Decoupled(flit)))
  })

  def wireLink(source: Vec[DecoupledIO[MeshFlit]], sink: Vec[DecoupledIO[MeshFlit]]): Unit = {
    val tx: Instance[MeshCreditTx] = Instantiate(new MeshCreditTx(p, linkDepth));
    val rx: Instance[MeshCreditRx] = Instantiate(new MeshCreditRx(p, linkDepth))
    tx.io.active := io.active; rx.io.active := io.active
    tx.io.in <> source; rx.io.link <> tx.io.link; sink <> rx.io.out
  }

  for {
    y <- 0 until p.yNodes
    x <- 0 until p.xNodes
  } {
    val router = routers(y)(x)
    for (vc <- 0 until p.virtualChannels) {
      val input  = io.localIn(index(x, y))(vc)
      val output = io.localOut(index(x, y))(vc)
      router.io.in(MeshDirection.Local)(vc).valid  := io.active && input.valid
      router.io.in(MeshDirection.Local)(vc).bits   := input.bits
      input.ready                                  := io.active && router.io.in(MeshDirection.Local)(vc).ready
      output.valid                                 := io.active && router.io.out(MeshDirection.Local)(vc).valid
      output.bits                                  := router.io.out(MeshDirection.Local)(vc).bits
      router.io.out(MeshDirection.Local)(vc).ready := io.active && output.ready
    }
    for (vc <- 0 until p.virtualChannels) {
      for (
        direction <- Seq(MeshDirection.West).filter(_ => x == 0) ++ Seq(MeshDirection.East).filter(_ =>
                       x == p.xNodes - 1
                     ) ++ Seq(MeshDirection.North).filter(_ => y == 0) ++ Seq(MeshDirection.South).filter(_ =>
                       y == p.yNodes - 1
                     )
      ) {
        router.io.in(direction)(vc).valid  := false.B; router.io.in(direction)(vc).bits := 0.U.asTypeOf(flit)
        router.io.out(direction)(vc).ready := false.B
      }
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
