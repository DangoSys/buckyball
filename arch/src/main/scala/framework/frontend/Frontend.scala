package framework.frontend

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.RoCCCommand
import freechips.rocketchip.tile.RoCCResponse
import framework.frontend.decoder.{GlobalDecoder, PostGDCmd}
import framework.frontend.globalrs.{
  GlobalReservationStation,
  GlobalReservationStationParam,
  GlobalRsComplete,
  GlobalRsIssue
}
import framework.frontend.FrontendParam

/**
 * Frontend Module
 * Encapsulates GlobalDecoder and GlobalReservationStation
 */
@instantiable
class Frontend(val parameter: FrontendParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[FrontendParam] {

  @public
  val io = IO(new Bundle {

    // RoCC command input
    val cmd = Flipped(Decoupled(new Bundle {
      val cmd = new RoCCCommand
    }))

    // Issue to domains
    val ball_issue_o = Decoupled(new GlobalRsIssue(parameter.rob_entries)(p))
    val mem_issue_o  = Decoupled(new GlobalRsIssue(parameter.rob_entries)(p))
    val gp_issue_o   = Decoupled(new GlobalRsIssue(parameter.rob_entries)(p))

    // Complete from domains
    val ball_complete_i = Flipped(Decoupled(new GlobalRsComplete(parameter.rob_entries)(p)))
    val mem_complete_i  = Flipped(Decoupled(new GlobalRsComplete(parameter.rob_entries)(p)))
    val gp_complete_i   = Flipped(Decoupled(new GlobalRsComplete(parameter.rob_entries)(p)))

    // RoCC response
    val resp = Decoupled(new RoCCResponse)
    val busy = Output(Bool())
  })

  // Instantiate GlobalDecoder
  val gDecoder: Instance[GlobalDecoder] = Instantiate(new GlobalDecoder(parameter))
  gDecoder.io.id_i.valid    := io.cmd.valid
  gDecoder.io.id_i.bits.cmd := io.cmd.bits.cmd
  io.cmd.ready              := gDecoder.io.id_i.ready

  // Instantiate GlobalReservationStation
  val globalRsParam = GlobalReservationStationParam(
    rob_entries = parameter.rob_entries,
    rs_out_of_order_response = parameter.rs_out_of_order_response
  )

  val globalRs: Instance[GlobalReservationStation] = Instantiate(new GlobalReservationStation(globalRsParam)(p))
  globalRs.io.global_decode_cmd_i <> gDecoder.io.id_o

  // Connect to domains
  io.ball_issue_o <> globalRs.io.ball_issue_o
  io.mem_issue_o <> globalRs.io.mem_issue_o
  io.gp_issue_o <> globalRs.io.gp_issue_o

  globalRs.io.ball_complete_i <> io.ball_complete_i
  globalRs.io.mem_complete_i <> io.mem_complete_i
  globalRs.io.gp_complete_i <> io.gp_complete_i

  // RoCC response
  io.resp <> globalRs.io.rs_rocc_o.resp
  io.busy := globalRs.io.rs_rocc_o.busy

  override lazy val desiredName = "Frontend"
}
