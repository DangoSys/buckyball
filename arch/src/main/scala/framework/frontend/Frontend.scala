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

  private val custom3Opcode = 0x7b
  private val custom3Funct3 = 3
  private val msetFunct     = 0x20
  private val mmioSetFunct  = 0x22
  private val initEndFunct  = 2

  private def bootCmd(funct: Int, rs1: BigInt, rs2: BigInt): RoCCCommandBB = {
    val cmd = Wire(new RoCCCommandBB(b.core.xLen))
    cmd         := 0.U.asTypeOf(cmd)
    cmd.funct   := funct.U
    cmd.funct3  := custom3Funct3.U
    cmd.opcode  := custom3Opcode.U
    cmd.rs1Data := rs1.U(b.core.xLen.W)
    cmd.rs2Data := rs2.U(b.core.xLen.W)
    cmd
  }

  private val bootBankCount = b.frontend.vbank_id_upper_bound + 1

  private val bootRecords =
    (0 until bootBankCount).map(bankId => bootCmd(msetFunct, bankId, 0)) ++
      (0 until b.memDomain.bankNum).map(bankId => bootCmd(mmioSetFunct, bankId, 0)) :+
      bootCmd(initEndFunct, 0, 0)

  private val bootRom     = VecInit(bootRecords)
  private val bootPcWidth = math.max(1, log2Ceil(bootRecords.length))
  // Phase-1 boot validation uses explicit software bb_boot_init(). Keep the
  // ROM path compiled but disabled until the explicit sequence is proven.
  val bootActive          = RegInit(false.B)
  val bootDrain           = RegInit(false.B)
  val bootWaitIdle        = RegInit(false.B)
  val bootPc              = RegInit(0.U(bootPcWidth.W))
  val bootCurrent         = bootRom(bootPc)
  val bootAtEnd           = bootCurrent.funct === initEndFunct.U
  val bootInjectValid     = bootActive && !bootDrain && !bootWaitIdle && !bootAtEnd

  when(bootActive && !bootDrain && bootAtEnd) {
    bootDrain := true.B
  }
  when(bootWaitIdle && scheduler.io.idle) {
    bootWaitIdle := false.B
  }
  when(bootDrain && scheduler.io.idle) {
    bootActive := false.B
  }
  when(bootInjectValid && gDecoder.io.id_i.ready) {
    bootPc       := bootPc + 1.U
    bootWaitIdle := true.B
  }

  gDecoder.io.id_i.valid    := Mux(bootActive, bootInjectValid, io.cmd.valid)
  gDecoder.io.id_i.bits.cmd := Mux(bootActive, bootCurrent, io.cmd.bits.cmd)
  io.cmd.ready              := !bootActive && gDecoder.io.id_i.ready

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
  io.busy := bootActive || scheduler.io.scheduler_rocc_o.busy

  // Barrier passthrough
  io.barrier_arrive            := scheduler.io.barrier_arrive
  scheduler.io.barrier_release := io.barrier_release

}
