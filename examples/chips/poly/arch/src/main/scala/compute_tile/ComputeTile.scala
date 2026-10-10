package examples.chips.poly.compute_tile

import chisel3.experimental.hierarchy.{instantiable, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.system.GlobalConfigOps._
import framework.system.core.CoreFactory
import framework.system.tile.{ComputeTilePlatform, TileConnection, TileEndpoint, TileFactory}

@instantiable
class ComputeTile(b: GlobalConfig) extends TileEndpoint(b.tileLinkParams, false) {
  val p               = b.tileParams
  val platform        = Instantiate(new ComputeTilePlatform(p))
  val controllerIndex = p.controller.get
  val controller      = CoreFactory.rocket(b.forCore(controllerIndex))
  val workers         = p.cores.indices.filterNot(_ == controllerIndex).map(i => CoreFactory.ant(b.forCore(i)))
  TileConnection.compute(platform, tlink, controller.clink, workers.map(_.clink))
}

object ComputeTile extends TileFactory {
  def instantiate(b: GlobalConfig): Instance[TileEndpoint] = Instantiate(new ComputeTile(b))
}
