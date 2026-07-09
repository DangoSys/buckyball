package framework.top

import upickle.default.{macroRW, ReadWriter}
import chisel3.experimental.SerializableModuleParameter
import framework.memdomain.configs.MemDomainParam
import framework.frontend.configs.FrontendParam
import framework.gpdomain.configs.GpDomainParam
import framework.balldomain.configs.BallDomainParam
import framework.system.core.configs.CoreParam
import framework.top.configs.TopConfig
import framework.system.core.configs.RocketCoreParam

case class GlobalConfig(
  memDomain:  MemDomainParam,
  frontend:   FrontendParam,
  gpDomain:   GpDomainParam,
  ballDomain: BallDomainParam,
  core:       CoreParam,
  top:        TopConfig,
  rocketCore: RocketCoreParam)
    extends SerializableModuleParameter

object GlobalConfig {
  implicit val rw: ReadWriter[GlobalConfig] = macroRW[GlobalConfig]

  def apply(): GlobalConfig = {
    GlobalConfig(
      memDomain = MemDomainParam(),
      frontend = FrontendParam(),
      gpDomain = GpDomainParam(),
      ballDomain = BallDomainParam(),
      core = CoreParam(),
      top = TopConfig(),
      rocketCore = RocketCoreParam()
    )
  }

}
