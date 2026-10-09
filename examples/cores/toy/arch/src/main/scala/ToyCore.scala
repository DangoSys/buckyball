package examples.cores.toy

import framework.top.GlobalConfig
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.system.core.{CoreConnection, RocketEndpoint}
import hier.core.rocket.Core
import framework.system.core.accelerator.BuckyballAccelerator

@instantiable
class ToyCore(b: GlobalConfig) extends RocketEndpoint(b) {
  require(p.buckyball.nonEmpty)
  val cpu         = Instantiate(new Core(p.data, p.instruction, p.regions, Some(p.commands))(p.cpu))
  clink.cpu <> cpu.clink
  val accelerator = Instantiate(new BuckyballAccelerator(p.buckyball.get))
  CoreConnection.accelerator(clink.accelerator.get, accelerator)
}

object ToyCore extends framework.system.core.RocketCoreFactory {
  def instantiate(b: GlobalConfig): Instance[RocketEndpoint] = Instantiate(new ToyCore(b))
}
