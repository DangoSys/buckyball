package framework.system.tile

import chisel3.experimental.hierarchy.{instantiable, Instantiate}
import framework.top.GlobalConfig
import framework.system.GlobalConfigOps._
import framework.system.core.CoreFactory

@instantiable
class MainTile(b: GlobalConfig) extends TileEndpoint(b.tileLinkParams, true) {
  require(b.tileDesign.get.design.factory.isEmpty, "The main tile is built into the device")
  val p        = b.tileParams
  val platform = Instantiate(new MainTilePlatform(p))
  val cores    = p.cores.indices.map(i => CoreFactory.rocket(b.forCore(i)))
  TileConnection.main(platform, tlink, cores.map(_.clink))
}
