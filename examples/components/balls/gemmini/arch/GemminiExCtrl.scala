package examples.balls.gemmini

import framework.acc.balldomain.configs.BallRouteParam

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.gemmini.configs.GemminiBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class GemminiExCtrl(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     GemminiBallParam)
    extends Module
    with GemminiExCtrlDefs
    with GemminiExCtrlDefaults
    with GemminiExCtrlCmdStates
    with GemminiExCtrlPreloadStates
    with GemminiExCtrlComputeReadState
    with GemminiExCtrlComputeFeedState
    with GemminiExCtrlStoreOps
    with GemminiExCtrlAccumulateState
    with GemminiExCtrlFsm {
  def commandDataBits: Int = frontendParam.commandDataBits
  @public
  val exio = io

  applyDefaults()
  runFsm()

  io.status.idle    := state === sIdle
  io.status.running := state =/= sIdle
}
