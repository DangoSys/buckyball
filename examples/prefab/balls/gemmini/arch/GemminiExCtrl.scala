package examples.balls.gemmini

import framework.generic.contract.blink.BlinkParam

import examples.balls.gemmini.configs.GemminiBallParam

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.generic.contract.configs.BallRouteParam

@instantiable
class GemminiExCtrl(
  val link:      BlinkParam,
  val instance:  BallRouteParam,
  val ballParam: GemminiBallParam)
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
  def commandDataBits: Int = link.dataBits
  @public
  val exio = io

  applyDefaults()
  runFsm()

  io.status.idle    := state === sIdle
  io.status.running := state =/= sIdle
}
