package examples.cores.goban

import framework.top.GlobalConfig
import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.ant.LocalExecution
import framework.system.core.{AntEndpoint, CoreConnection}
import framework.system.core.accelerator.{AntAdmission, BuckyballAccelerator}

@instantiable
class AntGobanCore(config: GlobalConfig) extends AntEndpoint(config) {
  val execution   = Instantiate(new LocalExecution(local))
  val accelerator = Instantiate(new BuckyballAccelerator(p.buckyball))
  val issuer      = Instantiate(new AntAdmission(local, tracking, cpuParams.core.nPMPs, bus))
  CoreConnection.ant(clink, execution, accelerator, issuer, signature)
}

object AntGobanCore extends framework.system.core.AntCoreFactory {
  def instantiate(b: GlobalConfig): Instance[AntEndpoint] = Instantiate(new AntGobanCore(b))
}
