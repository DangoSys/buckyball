package examples.balls.gemmini

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import framework.acc.balldomain.sub_rob.{Expansion, HasExpansion}
import buckyball.balls.gemmini.{Params => ParametersProto}
import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.gemmini.configs.GemminiBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.util._
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class GemminiBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink
    with HasExpansion {
  val ball = bp.balls.find(_.id == id).get

  val ballParam = GemminiBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  require(
    mp.bankEntries <= 1024,
    "GemminiBall rs2 row-base fields support at most 1024 bank entries"
  )

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  @public
  val expansion = IO(new Expansion(fp, mp, bp))

  val loop: Instance[Loop] = Instantiate(new Loop(fp, mp, bp, id))
  expansion <> loop.expansion

  val execution: Instance[GemminiExCtrl] =
    Instantiate(new GemminiExCtrl(fp, mp, bp, ball, ballParam))
  val robId = RegInit(0.U(log2Up(fp.rob_entries).W))
  when(blink.cmdReq.fire)(robId := blink.cmdReq.bits.rob_id)

  execution.exio.cmdReq <> blink.cmdReq
  blink.cmdResp <> execution.exio.cmdResp

  for (port <- 0 until ball.inBW) {
    blink.bankRead(port).io.req <> execution.exio.bankReadReq(port)
    execution.exio.bankReadResp(port) <> blink.bankRead(port).io.resp
    blink.bankRead(port).rob_id   := robId
    blink.bankRead(port).group_id := 0.U
  }
  blink.bankRead(0).bank_id := execution.exio.op1_bank_o
  blink.bankRead(0).group_id := execution.exio.op1_group_o
  if (ball.inBW > 1) {
    blink.bankRead(1).bank_id  := execution.exio.op2_bank_o
    blink.bankRead(1).group_id := execution.exio.op2_group_o
  }

  for (port <- 0 until ball.outBW) {
    blink.bankWrite(port).io <> execution.exio.bankWrite(port)
    blink.bankWrite(port).bank_id  := execution.exio.wr_bank_o
    blink.bankWrite(port).rob_id   := robId
    blink.bankWrite(port).group_id := execution.exio.wr_group_o + port.U
  }

  blink.status <> execution.exio.status
}

object GemminiBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts = {
    val ball = Instantiate(new GemminiBall(fp, mp, bp, id))
    BallPorts(ball.blink, Some(ball.expansion))
  }

}
