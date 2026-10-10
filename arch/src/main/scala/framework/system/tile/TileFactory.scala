package framework.system.tile

import chisel3.experimental.hierarchy.Instance
import framework.top.GlobalConfig

trait TileFactory {
  def instantiate(b: GlobalConfig): Instance[TileEndpoint]
}

object TileFactory {

  def instantiate(b: GlobalConfig): Instance[TileEndpoint] = {
    require(!b.tileDesign.get.design.main, "The main tile is built into the device")
    Class.forName(b.tileDesign.get.design.factory + "$")
      .getField("MODULE$").get(null).asInstanceOf[TileFactory].instantiate(b)
  }

}
