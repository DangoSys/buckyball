package framework.core.bbtile

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

import freechips.rocketchip.tilelink.{TLBundle, TLEdgeOut}
import framework.top.GlobalConfig
import framework.frontend.Frontend
import framework.gpdomain.GpDomain
import framework.memdomain.MemDomain
import framework.memdomain.frontend.outside_channel.tlb.{BBTLBExceptionIO, BBTLBPTWIO}
import examples.toy.balldomain.BallDomain

/**
 * Standalone Buckyball accelerator module.
 *
 * Decoupled from the LazyRoCCBB inheritance chain.
 * Uses @instantiable + GlobalConfig pattern.
 * TileLink bundles are passed in from the tile's diplomacy shell.
 *
 * @param b GlobalConfig for the accelerator
 * @param edge TLEdgeOut from the DMA TileLink nodes (for TLBundle sizing)
 */
@instantiable
class BuckyballAccelerator(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {
  val totalBallRead  = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val totalBallWrite = b.ballDomain.ballIdMappings.map(_.outBW).sum

  @public
  val io = IO(new Bundle {
    // RoCC command/response (connected to Rocket core inside tile)
    val cmd       = Flipped(Decoupled(new RoCCCommandBB(b.core.xLen)))
    val resp      = Decoupled(new RoCCResponseBB(b.core.xLen))
    val busy      = Output(Bool())
    val interrupt = Output(Bool())
    val hartid    = Input(UInt(b.core.xLen.W))

    // PTW interface (shared with Rocket core's PTW)
    val ptw    = Vec(1, new BBTLBPTWIO(b))
    // TLB exception interface
    val tlbExp = Vec(1, new BBTLBExceptionIO)
    // CPU sfence signal — flushes Buckyball's TLB
    val sfence = Input(Bool())

    // TileLink DMA bundles (from tile's diplomacy nodes)
    val tl_reader = new TLBundle(edge.bundle)
    val tl_writer = new TLBundle(edge.bundle)
  })

  // --- Instantiate domains ---
  val frontend:   Instance[Frontend]   = Instantiate(new Frontend(b))
  val ballDomain: Instance[BallDomain] = Instantiate(new BallDomain(b))
  val memDomain:  Instance[MemDomain]  = Instantiate(new MemDomain(b)(edge))
  val gpDomain:   Instance[GpDomain]   = Instantiate(new GpDomain(b))

  // --- Frontend <- cmd ---
  frontend.io.cmd.valid    := io.cmd.valid
  frontend.io.cmd.bits.cmd := io.cmd.bits
  io.cmd.ready             := frontend.io.cmd.ready

  // --- Frontend -> BallDomain ---
  ballDomain.global_issue_i <> frontend.io.ball_issue_o
  frontend.io.ball_complete_i <> ballDomain.global_complete_o

  // --- Frontend -> MemDomain ---
  memDomain.io.global_issue_i <> frontend.io.mem_issue_o
  frontend.io.mem_complete_i <> memDomain.io.global_complete_o
  memDomain.io.hartid := io.hartid

  // --- Frontend -> GpDomain ---
  gpDomain.io.global_issue_i <> frontend.io.gp_issue_o
  frontend.io.gp_complete_i <> gpDomain.io.global_complete_o

  // --- BallDomain <-> MemDomain (bankRead with pipeline register to break comb loops) ---
  for (i <- 0 until totalBallRead) {
    val bankReadReqWithIds = Wire(Decoupled(new Bundle {
      val bank_id  = chiselTypeOf(ballDomain.bankRead(i).bank_id)
      val rob_id   = chiselTypeOf(ballDomain.bankRead(i).rob_id)
      val ball_id  = chiselTypeOf(ballDomain.bankRead(i).ball_id)
      val group_id = chiselTypeOf(ballDomain.bankRead(i).group_id)
      val req      = chiselTypeOf(ballDomain.bankRead(i).io.req.bits)
    }))

    bankReadReqWithIds.valid            := ballDomain.bankRead(i).io.req.valid
    bankReadReqWithIds.bits.bank_id     := ballDomain.bankRead(i).bank_id
    bankReadReqWithIds.bits.rob_id      := ballDomain.bankRead(i).rob_id
    bankReadReqWithIds.bits.ball_id     := ballDomain.bankRead(i).ball_id
    bankReadReqWithIds.bits.group_id    := ballDomain.bankRead(i).group_id
    bankReadReqWithIds.bits.req         := ballDomain.bankRead(i).io.req.bits
    ballDomain.bankRead(i).io.req.ready := bankReadReqWithIds.ready

    val bankReadReqQ = Queue(bankReadReqWithIds, 8)

    memDomain.io.ballDomain.bankRead(i).io.req.valid := bankReadReqQ.valid
    memDomain.io.ballDomain.bankRead(i).io.req.bits  := bankReadReqQ.bits.req
    memDomain.io.ballDomain.bankRead(i).bank_id      := bankReadReqQ.bits.bank_id
    memDomain.io.ballDomain.bankRead(i).rob_id       := bankReadReqQ.bits.rob_id
    memDomain.io.ballDomain.bankRead(i).ball_id      := bankReadReqQ.bits.ball_id
    memDomain.io.ballDomain.bankRead(i).group_id     := bankReadReqQ.bits.group_id
    bankReadReqQ.ready                               := memDomain.io.ballDomain.bankRead(i).io.req.ready

    ballDomain.bankRead(i).io.resp <> memDomain.io.ballDomain.bankRead(i).io.resp
  }

  ballDomain.bankWrite <> memDomain.io.ballDomain.bankWrite

  // --- PTW ---
  io.ptw(0).req <> memDomain.io.ptw(0).req
  memDomain.io.ptw(0).resp <> io.ptw(0).resp
  memDomain.io.ptw(0).ptbr <> io.ptw(0).ptbr
  memDomain.io.ptw(0).hgatp <> io.ptw(0).hgatp
  memDomain.io.ptw(0).vsatp <> io.ptw(0).vsatp
  memDomain.io.ptw(0).status <> io.ptw(0).status
  memDomain.io.ptw(0).hstatus <> io.ptw(0).hstatus
  memDomain.io.ptw(0).gstatus <> io.ptw(0).gstatus
  memDomain.io.ptw(0).pmp <> io.ptw(0).pmp
  memDomain.io.ptw(0).customCSRs := DontCare

  // --- TLB exception ---
  memDomain.io.tlbExp(0).flush_skip  := false.B
  memDomain.io.tlbExp(0).flush_retry := io.sfence
  io.tlbExp(0) <> memDomain.io.tlbExp(0)

  // --- TileLink DMA ---
  io.tl_reader <> memDomain.io.tl_reader
  io.tl_writer <> memDomain.io.tl_writer

  // --- Response & status ---
  io.resp <> frontend.io.resp
  io.busy      := frontend.io.busy
  io.interrupt := memDomain.io.tlbExp(0).interrupt

  // --- Busy watchdog ---
  val busy_counter = RegInit(0.U(32.W))
  busy_counter := Mux(frontend.io.busy, busy_counter + 1.U, 0.U)
  assert(busy_counter < 10000.U, "BuckyballAccelerator: busy for too long!")
}
