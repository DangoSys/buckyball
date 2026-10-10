package sims.verilator

import chisel3.experimental.hierarchy.{Instance, Instantiate}
import framework.system.{System, SystemParams}

import sims.soc.SystemTarget

/** Poly on the explicit System: a CPU main tile and two homogeneous compute tiles. */
class PolyVerilatorTarget extends SystemTarget("../examples/chips/poly/configs/generated/chip.pb") {
  override def instantiate(p: SystemParams): Instance[System] =
    Instantiate(new framework.system.Device(p.toGlobalConfig))
  override val dramBytes: BigInt = BigInt(16) << 30
}
