package examples.balls.vector

import framework.acc.balldomain.configs.BallRouteParam

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}
import examples.balls.vector.configs.VectorBallParam

@instantiable
class VecUnit(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     VectorBallParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val cfg             = ballParam
  val lane            = cfg.lane
  val inW             = cfg.inputWidth
  val accW            = cfg.outputWidth

  require(lane == 16, "VecUnit vecmat16 requires lane=16")
  require(inW == 8, "VecUnit vecmat16 requires int8 inputs")
  require(accW == 32, "VecUnit vecmat16 requires int32 accumulators")

  val map = instance

  val inBW  = map.inBW
  val outBW = map.outBW
  require(inBW >= 2, "VecUnit vecmat16 requires two read ports")
  require(outBW >= 4, "VecUnit vecmat16 requires four write ports")

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      inBW,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val bankWrite = Vec(
      outBW,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val status = new BallStatus
  })

  val thread =
    Instantiate(new VecMat16(frontendParam, memParam, routingParam, inBW, outBW, inW, accW, lane, ballParam))

  thread.io.cmdReq <> io.cmdReq
  thread.io.cmdResp <> io.cmdResp
  for (i <- 0 until inBW) { thread.io.bankRead(i) <> io.bankRead(i) }
  for (i <- 0 until outBW) { thread.io.bankWrite(i) <> io.bankWrite(i) }
  io.status <> thread.io.status
}
