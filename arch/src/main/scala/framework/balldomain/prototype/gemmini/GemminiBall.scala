package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.top.GlobalConfig

@instantiable
class GemminiBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  val ballCommonConfig = b.ballDomain.ballIdMappings.find(_.ballName == "GemminiBall")
    .getOrElse(throw new IllegalArgumentException("GemminiBall not found in config"))
  val inBW             = ballCommonConfig.inBW
  val outBW            = ballCommonConfig.outBW

  @public
  val io = IO(new BlinkIO(b, inBW, outBW))

  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  val exCtrl: Instance[GemminiExCtrl] = Instantiate(new GemminiExCtrl(b))

  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  // Command interface
  exCtrl.io.cmdReq <> io.cmdReq
  exCtrl.io.cmdResp <> io.cmdResp

  // Bank read ports
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req <> exCtrl.io.bankReadReq(i)
    exCtrl.io.bankReadResp(i) <> io.bankRead(i).io.resp
    io.bankRead(i).rob_id   := rob_id_reg
    io.bankRead(i).ball_id  := 0.U
    io.bankRead(i).group_id := 0.U
  }
  io.bankRead(0).bank_id := exCtrl.io.op1_bank_o
  if (inBW > 1) {
    io.bankRead(1).bank_id := exCtrl.io.op2_bank_o
  }

  // Bank write ports
  for (i <- 0 until outBW) {
    io.bankWrite(i).io <> exCtrl.io.bankWrite(i)
    io.bankWrite(i).bank_id  := exCtrl.io.wr_bank_o
    io.bankWrite(i).rob_id   := rob_id_reg
    io.bankWrite(i).ball_id  := 0.U
    io.bankWrite(i).group_id := i.U
  }

  io.status <> exCtrl.io.status

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
}
