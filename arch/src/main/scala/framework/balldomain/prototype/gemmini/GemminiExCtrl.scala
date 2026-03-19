package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class GemminiExCtrl(val b: GlobalConfig)
    extends Module
    with GemminiExCtrlDefs
    with GemminiExCtrlDefaults
    with GemminiExCtrlCmdStates
    with GemminiExCtrlPreloadStates
    with GemminiExCtrlComputeReadState
    with GemminiExCtrlComputeFeedState
    with GemminiExCtrlStoreOps
    with GemminiExCtrlFsm {
  @public val exio = io

  applyDefaults()
  runFsm()

  io.status.idle    := state === sIdle
  io.status.running := state =/= sIdle
}
