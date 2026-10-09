package framework.top

import upickle.default.{macroRW, ReadWriter}
import chisel3.experimental.SerializableModuleParameter
import framework.memdomain.configs.MemDomainParam
import framework.frontend.configs.FrontendParam
import framework.rvv.configs.RvvParam
import framework.balldomain.configs.BallDomainParam
import framework.system.tile.configs.TileParam
import framework.top.configs.SimParam

case class GlobalConfig(
  memDomain:     MemDomainParam,
  frontend:      FrontendParam,
  rvv:           RvvParam,
  ballDomain:    BallDomainParam,
  tile:          TileParam,
  sim:           SimParam,
  coreSignature: BigInt = 0,
  systemDesign:  Option[framework.top.configs.SystemDesign] = None,
  tileDesign:    Option[framework.top.configs.TileScope] = None,
  coreDesign:    Option[framework.top.configs.CoreScope] = None)
    extends SerializableModuleParameter

object GlobalConfig {

  def apply(): GlobalConfig = {
    GlobalConfig(
      memDomain = MemDomainParam(),
      frontend = FrontendParam(),
      rvv = RvvParam(),
      ballDomain = BallDomainParam(),
      tile = TileParam(),
      sim = SimParam()
    )
  }

  implicit val rw: ReadWriter[GlobalConfig] = macroRW[GlobalConfig]

}
