package framework.system.core

import framework.ant.{Params => AntParams}
import framework.system.core.rocket.CpuParams
import framework.top.GlobalConfig
import memcore.bus.chi
import memcore.memory.interlock.{Params => TrackingParams}

case class AntCoreParams(
  buckyball: GlobalConfig,
  local:     AntParams,
  tracking:  TrackingParams,
  bus:       chi.Params,
  signature: BigInt,
  cpu:       CpuParams)
