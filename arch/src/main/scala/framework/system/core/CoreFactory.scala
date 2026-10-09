package framework.system.core

import chisel3.experimental.hierarchy.Instance
import framework.top.GlobalConfig

trait RocketCoreFactory {
  def instantiate(b: GlobalConfig): Instance[RocketEndpoint]
}

trait AntCoreFactory {
  def instantiate(b: GlobalConfig): Instance[AntEndpoint]
}

object CoreFactory {

  def rocket(b: GlobalConfig): Instance[RocketEndpoint] =
    Class.forName(b.coreDesign.get.factory + "$")
      .getField("MODULE$").get(null).asInstanceOf[RocketCoreFactory].instantiate(b)

  def ant(b: GlobalConfig): Instance[AntEndpoint] =
    Class.forName(b.coreDesign.get.factory + "$")
      .getField("MODULE$").get(null).asInstanceOf[AntCoreFactory].instantiate(b)

}
