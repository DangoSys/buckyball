package framework.frontend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.frontend.decoder.{GlobalDecoder, PostGDCmd}
import framework.frontend.globalrs.{GlobalSchedComplete, GlobalSchedIssue, GlobalScheduler}
import framework.top.GlobalConfig
import framework.system.core.rocket.{RoCCCommandBB, RoCCResponseBB}
import framework.balldomain.blink.SubRobRow

/**
 * Frontend Module
 * Encapsulates GlobalDecoder and global scheduler
 */
@instantiable
class Frontend(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {

    // RoCC command input
    val cmd = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommandBB(b.core.xLen)
    }))

    // Issue to domains
    val ball_issue_o    = Decoupled(new GlobalSchedIssue(b))
    val mem_issue_o     = Decoupled(new GlobalSchedIssue(b))
    val gp_issue_o      = Decoupled(new GlobalSchedIssue(b))
    // Complete from domains
    val ball_complete_i = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val mem_complete_i  = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val gp_complete_i   = Flipped(Decoupled(new GlobalSchedComplete(b)))

    // Ball -> SubROB request passthrough
    val ball_subrob_req_i = Flipped(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))

    // RoCC response
    val resp = Decoupled(new RoCCResponseBB(b.core.xLen))
    val busy = Output(Bool())

    // Barrier interface — passthrough to GlobalRS
    val barrier_arrive  = Output(Bool())
    val barrier_release = Input(Bool())
  })

  val gDecoder:  Instance[GlobalDecoder]   = Instantiate(new GlobalDecoder(b))
  val scheduler: Instance[GlobalScheduler] = Instantiate(new GlobalScheduler(b))

  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits.cmd
  io.cmd.ready              := gDecoder.io.id_i.ready

  scheduler.io.decode_cmd_i <> gDecoder.io.id_o

  io.ball_issue_o <> scheduler.io.ball_issue_o
  io.mem_issue_o <> scheduler.io.mem_issue_o
  io.gp_issue_o <> scheduler.io.gp_issue_o

  scheduler.io.ball_complete_i <> io.ball_complete_i
  scheduler.io.mem_complete_i <> io.mem_complete_i
  scheduler.io.gp_complete_i <> io.gp_complete_i

  // Wire SubROB request from BallDomain through to scheduler
  for (i <- 0 until b.ballDomain.ballNum) {
    scheduler.io.ball_subrob_req_i(i) <> io.ball_subrob_req_i(i)
  }

  io.resp <> scheduler.io.scheduler_rocc_o.resp
  io.busy := scheduler.io.scheduler_rocc_o.busy

  // Barrier passthrough
  io.barrier_arrive            := scheduler.io.barrier_arrive
  scheduler.io.barrier_release := io.barrier_release

}
