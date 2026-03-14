package framework.balldomain.blink

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.frontend.decoder.PostGDCmd
import framework.frontend.scoreboard.BankAccessInfo

class SubRobSlot(val b: GlobalConfig) extends Bundle {
  val valid = Bool()
  val cmd   = new PostGDCmd(b)
}

class SubRobRow(val b: GlobalConfig) extends Bundle {
  val slots         = Vec(4, new SubRobSlot(b))
  val ball_id       = UInt(log2Up(b.ballDomain.ballNum).W)
  val master_rob_id = UInt(log2Up(b.frontend.rob_entries).W)
}

/** Tie-off for subRobReq when a Ball does not use SubROB. */
object SubRobRow {

  def tieOff(b: GlobalConfig): SubRobRow = {
    val w = Wire(new SubRobRow(b))
    w.ball_id       := 0.U
    w.master_rob_id := 0.U
    val bankIdLen = log2Up(b.memDomain.bankNum)
    for (i <- 0 until 4) {
      w.slots(i).valid            := false.B
      w.slots(i).cmd.domain_id    := 0.U
      w.slots(i).cmd.cmd.raw_inst := 0.U
      w.slots(i).cmd.cmd.funct    := 0.U
      w.slots(i).cmd.cmd.rs2      := 0.U
      w.slots(i).cmd.cmd.rs1      := 0.U
      w.slots(i).cmd.cmd.xd       := false.B
      w.slots(i).cmd.cmd.xs1      := false.B
      w.slots(i).cmd.cmd.xs2      := false.B
      w.slots(i).cmd.cmd.rd       := 0.U
      w.slots(i).cmd.cmd.opcode   := 0.U
      w.slots(i).cmd.cmd.rs1Data  := 0.U
      w.slots(i).cmd.cmd.rs2Data  := 0.U
      w.slots(i).cmd.bankAccess   := BankAccessInfo.none(bankIdLen)
      w.slots(i).cmd.isFence      := false.B
      w.slots(i).cmd.isBarrier    := false.B
    }
    w
  }

}
