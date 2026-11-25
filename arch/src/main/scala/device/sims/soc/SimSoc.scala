package framework.device.sims.soc

import chisel3.experimental.hierarchy.public

import chisel3.experimental.hierarchy.instantiable

import framework.device.sims.configs.GlobalConfig

import chisel3._
import chisel3.experimental.hierarchy.Instance
import framework.device.dlink.BaseDevice
import framework.device.configs.DeviceParam
import framework.generic.bus.axi4
import framework.generic.contract.sims.configs.SimParam

/** A design supplies its parameters and explicitly instantiates its DLink implementation. */
abstract class SystemTarget(val b: GlobalConfig) {
  def instantiate(b: GlobalConfig): Instance[BaseDevice]
  // Modules Verilator builds once as hierarchical blocks, however many instances: the design's large repeated units.
  def hierBlocks: Seq[String]
  val dramBase:  BigInt = b.device.ddr.base
  val dramBytes: BigInt = b.device.ddr.bytes
}

@instantiable
class SimSoc(b: GlobalConfig, target: SystemTarget) extends Module {
  val diffTest = b.sim.diffTest

  val p = b.device

  @public
  val io = IO(new Bundle {

    val axi = new axi4.Port(p.ddr.port)

  })

  val system: Instance[BaseDevice] = target.instantiate(b)
  io.axi <> system.dlink.axi
}
