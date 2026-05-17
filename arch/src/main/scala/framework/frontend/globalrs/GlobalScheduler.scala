package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.{DomainId, PostGDCmd}
import framework.frontend.decoder.GISA._
import framework.frontend.scoreboard.BankAccessInfo
import framework.system.core.rocket.RoCCResponseBB
import framework.balldomain.blink.SubRobRow

class GlobalRobEntry(val b: GlobalConfig) extends Bundle {
  val cmd               = new PostGDCmd(b)
  val renamedBankAccess = new BankAccessInfo(b.frontend.bank_id_len)
  val rob_id            = UInt(log2Up(b.frontend.rob_entries).W)
}

class GlobalSchedIssue(b: GlobalConfig) extends GlobalRobEntry(b) {
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

class GlobalSchedComplete(b: GlobalConfig) extends Bundle {
  val rob_id     = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub     = Bool()
  val sub_rob_id = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
}

@instantiable
class GlobalScheduler(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val decode_cmd_i      = Flipped(new DecoupledIO(new PostGDCmd(b)))
    val ball_issue_o      = Decoupled(new GlobalSchedIssue(b))
    val mem_issue_o       = Decoupled(new GlobalSchedIssue(b))
    val gp_issue_o        = Decoupled(new GlobalSchedIssue(b))
    val ball_complete_i   = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val mem_complete_i    = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val gp_complete_i     = Flipped(Decoupled(new GlobalSchedComplete(b)))
    val ball_subrob_req_i = Flipped(Vec(b.ballDomain.ballNum, Decoupled(new SubRobRow(b))))

    val scheduler_rocc_o = new Bundle {
      val resp = new DecoupledIO(new RoCCResponseBB(b.core.xLen))
      val busy = Output(Bool())
    }

    val barrier_arrive  = Output(Bool())
    val barrier_release = Input(Bool())
  })

  val rob:    Instance[GlobalROB] = Instantiate(new GlobalROB(b))
  val subRob: Instance[SubROB]    = Instantiate(new SubROB(b))

  val isFenceCmd  = io.decode_cmd_i.valid && io.decode_cmd_i.bits.isFence
  val fenceActive = RegInit(false.B)
  when(isFenceCmd && !fenceActive) {
    fenceActive := true.B
  }
  when(fenceActive && rob.io.empty) {
    fenceActive := false.B
  }

  val isBarrierCmd       = io.decode_cmd_i.valid && io.decode_cmd_i.bits.isBarrier
  val barrierWaitROB     = RegInit(false.B)
  val barrierWaitRelease = RegInit(false.B)
  when(isBarrierCmd && !barrierWaitROB && !barrierWaitRelease && !fenceActive) {
    barrierWaitROB := true.B
  }
  when(barrierWaitROB && rob.io.empty) {
    barrierWaitROB     := false.B
    barrierWaitRelease := true.B
  }
  when(barrierWaitRelease && io.barrier_release) {
    barrierWaitRelease := false.B
  }
  io.barrier_arrive := barrierWaitRelease

  val isFrontendCmd = io.decode_cmd_i.bits.isFence || io.decode_cmd_i.bits.isBarrier
  val anyStall      = fenceActive || barrierWaitROB || barrierWaitRelease
  rob.io.alloc.valid    := io.decode_cmd_i.valid && !isFrontendCmd && !anyStall
  rob.io.alloc.bits     := io.decode_cmd_i.bits
  io.decode_cmd_i.ready := Mux(
    isFrontendCmd,
    !anyStall,
    rob.io.alloc.ready && !anyStall
  )

  val subRobWriteArb = Module(new Arbiter(new SubRobRow(b), b.ballDomain.ballNum))
  for (i <- 0 until b.ballDomain.ballNum) {
    subRobWriteArb.io.in(i) <> io.ball_subrob_req_i(i)
  }
  subRob.io.write <> subRobWriteArb.io.out

  val is_ball_domain = rob.io.issue.bits.cmd.domain_id === DomainId.BALL
  val is_mem_domain  = rob.io.issue.bits.cmd.domain_id === DomainId.MEM
  val is_gp_domain   = rob.io.issue.bits.cmd.domain_id === DomainId.GP

  val subRobIssueValid = subRob.io.issue.valid
  val subRobCmd        = subRob.io.issue.bits

  val subRobIssueEntry = Wire(new GlobalSchedIssue(b))
  subRobIssueEntry.cmd               := subRobCmd
  subRobIssueEntry.renamedBankAccess := 0.U.asTypeOf(subRobIssueEntry.renamedBankAccess)
  subRobIssueEntry.rob_id            := subRob.io.issueMasterRobId
  subRobIssueEntry.is_sub            := true.B
  subRobIssueEntry.sub_rob_id        := subRob.io.issueSubId

  val subRobIssBall = subRobCmd.domain_id === DomainId.BALL
  val subRobIssMem  = subRobCmd.domain_id === DomainId.MEM
  val subRobIssGp   = subRobCmd.domain_id === DomainId.GP

  val mainIssueEntry = Wire(new GlobalSchedIssue(b))
  mainIssueEntry.cmd               := rob.io.issue.bits.cmd
  mainIssueEntry.renamedBankAccess := rob.io.issue.bits.renamedBankAccess
  mainIssueEntry.rob_id            := rob.io.issue.bits.rob_id
  mainIssueEntry.is_sub            := false.B
  mainIssueEntry.sub_rob_id        := 0.U

  io.ball_issue_o.valid := Mux(
    subRobIssueValid && subRobIssBall,
    true.B,
    rob.io.issue.valid && is_ball_domain && !subRobIssueValid
  )
  io.ball_issue_o.bits  := Mux(subRobIssueValid && subRobIssBall, subRobIssueEntry, mainIssueEntry)

  io.mem_issue_o.valid := Mux(
    subRobIssueValid && subRobIssMem,
    true.B,
    rob.io.issue.valid && is_mem_domain && !subRobIssueValid
  )
  io.mem_issue_o.bits  := Mux(subRobIssueValid && subRobIssMem, subRobIssueEntry, mainIssueEntry)

  io.gp_issue_o.valid := Mux(
    subRobIssueValid && subRobIssGp,
    true.B,
    rob.io.issue.valid && is_gp_domain && !subRobIssueValid
  )
  io.gp_issue_o.bits  := Mux(subRobIssueValid && subRobIssGp, subRobIssueEntry, mainIssueEntry)

  subRob.io.issue.ready :=
    (subRobIssBall && io.ball_issue_o.ready) ||
      (subRobIssMem && io.mem_issue_o.ready) ||
      (subRobIssGp && io.gp_issue_o.ready)

  rob.io.issue.ready  := !subRobIssueValid && (
    (is_ball_domain && io.ball_issue_o.ready) ||
      (is_mem_domain && io.mem_issue_o.ready) ||
      (is_gp_domain && io.gp_issue_o.ready)
  )
  rob.io.subRobActive := subRobIssueValid

  val completeArb = Module(new Arbiter(new GlobalSchedComplete(b), 3))
  completeArb.io.in(0).valid := io.ball_complete_i.valid
  completeArb.io.in(0).bits  := io.ball_complete_i.bits
  io.ball_complete_i.ready   := completeArb.io.in(0).ready
  completeArb.io.in(1).valid := io.mem_complete_i.valid
  completeArb.io.in(1).bits  := io.mem_complete_i.bits
  io.mem_complete_i.ready    := completeArb.io.in(1).ready
  completeArb.io.in(2).valid := io.gp_complete_i.valid
  completeArb.io.in(2).bits  := io.gp_complete_i.bits
  io.gp_complete_i.ready     := completeArb.io.in(2).ready

  val completeBits = completeArb.io.out.bits
  subRob.io.subComplete.valid    := completeArb.io.out.valid && completeBits.is_sub
  subRob.io.subComplete.bits     := completeBits.sub_rob_id
  subRob.io.masterComplete.ready := true.B

  val normalComplete = completeArb.io.out.valid && !completeBits.is_sub
  if (b.frontend.rs_out_of_order_response) {
    rob.io.complete.valid := normalComplete || subRob.io.masterComplete.valid
    rob.io.complete.bits  := Mux(subRob.io.masterComplete.valid, subRob.io.masterComplete.bits, completeBits.rob_id)
  } else {
    val isHeadComplete = Mux(
      subRob.io.masterComplete.valid,
      subRob.io.masterComplete.bits === rob.io.head_ptr,
      completeBits.rob_id === rob.io.head_ptr
    )
    rob.io.complete.valid := (normalComplete || subRob.io.masterComplete.valid) && isHeadComplete
    rob.io.complete.bits  := Mux(subRob.io.masterComplete.valid, subRob.io.masterComplete.bits, completeBits.rob_id)
  }
  completeArb.io.out.ready := Mux(
    completeBits.is_sub,
    subRob.io.subComplete.ready,
    rob.io.complete.ready
  )

  io.scheduler_rocc_o.resp.valid     := false.B
  io.scheduler_rocc_o.resp.bits.rd   := 0.U
  io.scheduler_rocc_o.resp.bits.data := 0.U
  io.scheduler_rocc_o.busy           := rob.io.full || fenceActive || barrierWaitROB || barrierWaitRelease
}
