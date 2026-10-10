package framework.system

import chisel3.experimental.hierarchy.{instantiable, Instantiate}
import framework.top.GlobalConfig
import framework.system.GlobalConfigOps._
import framework.system.tile.{MainTile, TileFactory}

@instantiable
class Device(b: GlobalConfig) extends System(b.systemParams) {
  val main  = Instantiate(new MainTile(b.forTile(0)))
  val tiles = b.systemDesign.get.tiles.indices.drop(1).map(i => TileFactory.instantiate(b.forTile(i)))
  DeviceConnection.connect(b.systemParams, dlink, Seq(main.tlink) ++ tiles.map(_.tlink))
}
