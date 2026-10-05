package framework.system.core.rocket

import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, RocketCoreParams}

/** Explicit CPU metadata; the upstream parameter conversion stays in the Rocket adapter. */
case class CpuParams(
  core:                RocketCoreParams,
  dcache:              Option[DCacheParams],
  icache:              Option[ICacheParams],
  btb:                 Option[BTBParams],
  physicalAddressBits: Int,
  hartIdBits:          Int,
  beatBytes:           Int = 8,
  blockBytes:          Int = 64,
  asIdBits:            Int = 0)
