package framework.frontend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.frontend.decoder.{GlobalDecoder, PostGDCmd}
import framework.frontend.globalrs.{GlobalReservationStation, GlobalRsComplete, GlobalRsIssue}
import framework.top.GlobalConfig
import framework.core.rocket.{RoCCCommandBB, RoCCResponseBB}

/**
 * Frontend Module
 * Encapsulates GlobalDecoder and GlobalReservationStation
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
    val ball_issue_o    = Decoupled(new GlobalRsIssue(b))
    val mem_issue_o     = Decoupled(new GlobalRsIssue(b))
    val gp_issue_o      = Decoupled(new GlobalRsIssue(b))
    // Complete from domains
    val ball_complete_i = Flipped(Decoupled(new GlobalRsComplete(b)))
    val mem_complete_i  = Flipped(Decoupled(new GlobalRsComplete(b)))
    val gp_complete_i   = Flipped(Decoupled(new GlobalRsComplete(b)))

    // RoCC response
    val resp = Decoupled(new RoCCResponseBB(b.core.xLen))
    val busy = Output(Bool())
  })

  val gDecoder: Instance[GlobalDecoder]            = Instantiate(new GlobalDecoder(b))
  val globalRs: Instance[GlobalReservationStation] = Instantiate(new GlobalReservationStation(b))

  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits.cmd
  io.cmd.ready              := gDecoder.io.id_i.ready

  globalRs.io.global_decode_cmd_i <> gDecoder.io.id_o

  io.ball_issue_o <> globalRs.io.ball_issue_o
  io.mem_issue_o <> globalRs.io.mem_issue_o
  io.gp_issue_o <> globalRs.io.gp_issue_o

  globalRs.io.ball_complete_i <> io.ball_complete_i
  globalRs.io.mem_complete_i <> io.mem_complete_i
  globalRs.io.gp_complete_i <> io.gp_complete_i

  io.resp <> globalRs.io.rs_rocc_o.resp
  io.busy := globalRs.io.rs_rocc_o.busy

}
