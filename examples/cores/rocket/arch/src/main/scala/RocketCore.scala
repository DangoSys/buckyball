package examples.cores.rocket

import framework.top.GlobalConfig
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.system.core.{CoreConnection, RocketEndpoint}
import hier.core.rocket.Core

@instantiable
class RocketCore(b: GlobalConfig) extends RocketEndpoint(b) {
  require(p.buckyball.isEmpty)
  val cpu = Instantiate(new Core(p.data, p.instruction, p.regions, Some(p.commands))(p.cpu))
  clink.cpu <> cpu.clink
}

object RocketCore extends framework.system.core.RocketCoreFactory {
  def instantiate(b: GlobalConfig): Instance[RocketEndpoint] = Instantiate(new RocketCore(b))
}
