package framework.top

import upickle.default.{macroRW, ReadWriter}
import chisel3.experimental.SerializableModuleParameter
import framework.memdomain.configs.MemDomainParam
import framework.frontend.configs.FrontendParam
import framework.gpdomain.configs.GpDomainParam
import framework.balldomain.configs.BallDomainParam
import framework.system.tile.configs.TileParam
import framework.top.configs.SimParam

case class GlobalConfig(
  memDomain:  MemDomainParam,
  frontend:   FrontendParam,
  gpDomain:   GpDomainParam,
  ballDomain: BallDomainParam,
  tile:       TileParam,
  sim:        SimParam)
    extends SerializableModuleParameter

object GlobalConfig {
  implicit val rw: ReadWriter[GlobalConfig] = macroRW[GlobalConfig]

  def apply(): GlobalConfig = {
    GlobalConfig(
      memDomain = MemDomainParam(),
      frontend = FrontendParam(),
      gpDomain = GpDomainParam(),
      ballDomain = BallDomainParam(),
      tile = TileParam(),
      sim = SimParam()
    )
  }

}
