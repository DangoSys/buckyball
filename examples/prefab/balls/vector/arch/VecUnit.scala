package examples.balls.vector

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.blink.BallResponse

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.ball.blink.BallStatus
import examples.balls.vector.configs.VectorBallParam
import framework.generic.contract.configs.BallRouteParam
import framework.generic.contract.blink.BallCommand
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite

@instantiable
class VecUnit(
  link:          BlinkParam,
  instance:      BallRouteParam,
  val ballParam: VectorBallParam)
    extends Module {
  val commandDataBits = link.dataBits
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

    val cmdReq = Flipped(Decoupled(new BallCommand(link)))

    val cmdResp = Decoupled(new BallResponse(link))

    val bankRead = Vec(
      inBW,
      Flipped(new BankRead(
        link
      ))
    )

    val bankWrite = Vec(
      outBW,
      Flipped(new BankWrite(
        link
      ))
    )

    val status = new BallStatus
  })

  val thread =
    Instantiate(new VecMat16(link, inBW, outBW, inW, accW, lane, ballParam))

  thread.io.cmdReq <> io.cmdReq
  thread.io.cmdResp <> io.cmdResp
  for (i <- 0 until inBW) { thread.io.bankRead(i) <> io.bankRead(i) }
  for (i <- 0 until outBW) { thread.io.bankWrite(i) <> io.bankWrite(i) }
  io.status <> thread.io.status
}
