package examples.balls.gemmini

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import framework.ball.sub_rob.{Expansion, HasExpansion}
import buckyball.balls.gemmini.{Params => ParametersProto}

import examples.balls.gemmini.configs.GemminiBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import chisel3.util._
import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class GemminiBall(
  link:      BlinkParam,
  route:     BallRouteParam,
  expanding: ExpansionParam)
    extends Module
    with HasBlink
    with HasExpansion {
  val ball = route

  val ballParam = GemminiBallParam.fromProto(ball.parameters.unpack(classOf[ParametersProto]))

  require(
    link.bankEntries <= 1024,
    "GemminiBall rs2 row-base fields support at most 1024 bank entries"
  )

  @public
  val blink = IO(new BlinkIO(link, route))

  @public
  val expansion = IO(new Expansion(expanding.host, expanding.balls))

  val loop: Instance[Loop] = Instantiate(new Loop(expanding.host, expanding.balls, route.id))
  expansion <> loop.expansion

  val execution: Instance[GemminiExCtrl] =
    Instantiate(new GemminiExCtrl(link, ball, ballParam))
  val robId = RegInit(0.U(link.robIdBits.W))
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
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts = {
    val ball = Instantiate(new GemminiBall(link, route, expansion.get))
    BallPorts(ball.blink, Some(ball.expansion))
  }

}
