package framework.system.tile

import chisel3._
import chisel3.experimental.hierarchy.Instance
import framework.system.core.RocketCLink
import framework.system.core.clink.CLinkIO
import framework.system.tile.tlink.TLinkIO

/** Cross-level binding is shared by all tile designs. */
object TileConnection {

  def main(platform: Instance[MainTilePlatform], link: TLinkIO, cores: Seq[RocketCLink]): Unit = {
    require(cores.size == platform.coreLinks.size)
    cores.zipWithIndex.foreach { case (core, i) => platform.coreLinks(i) <> core }
    link <> platform.tlink
  }

  def compute(
    platform:   Instance[ComputeTilePlatform],
    link:       TLinkIO,
    controller: RocketCLink,
    workers:    Seq[CLinkIO]
  ): Unit = {
    require(workers.size == platform.workerLinks.size)
    platform.controllerLink <> controller
    workers.zipWithIndex.foreach { case (core, i) => platform.workerLinks(i) <> core }
    link <> platform.tlink
  }

}
