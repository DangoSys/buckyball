package framework.system.tile

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public, Instance}
import framework.system.configloader.{AntTileCore, RocketTileCore, SharedStorageParams, TileCore}
import framework.system.core.rocket.CpuParams
import framework.system.core.{AntCoreParams, RocketCoreParams}
import hier.core.rocket.Commands
import framework.system.tile.tlink.{HasTLink, TLinkIO}
import memcore.bus.axi4
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.interlock.{Params => TrackingParams}

case class TileLinkParams(
  main:          Boolean,
  cpus:          Int,
  executions:    Int,
  controls:      Int,
  bus:           memcore.bus.chi.Params,
  axi:           axi4.Params,
  sharedStorage: Boolean)

/** Structural parameters only; tile IDs and hart IDs arrive through TLink. */
case class TileParams(
  main:          Boolean,
  cores:         Seq[TileCore],
  signatures:    Seq[BigInt],
  controller:    Option[Int],
  cpus:          Seq[CpuParams],
  memory:        CoherenceParams,
  l1:            RnfParams,
  regions:       Seq[PhysicalRegion],
  tracking:      TrackingParams,
  controls:      Int,
  axi:           axi4.Params,
  sharedStorage: Option[SharedStorageParams],
  tiles:         Int) {

  def linkParams: TileLinkParams =
    TileLinkParams(main, cpus.size, cores.size, if (main) controls else 0, memory.chi, axi, sharedStorage.isDefined)

  def rocket(index: Int): RocketCoreParams = {
    val core     = cores(index) match {
      case value: RocketTileCore => value
      case _ => throw new IllegalArgumentException("Rocket parameters require a Rocket slot")
    }
    val cpuIndex = cores.take(index).count(_.isInstanceOf[RocketTileCore])
    RocketCoreParams(
      cpus(cpuIndex),
      l1.copy(nodeId = cpuIndex + 1),
      l1.copy(nodeId = cpus.size + cpuIndex + 1),
      regions,
      Commands(core.buckyball.isDefined, true, tracking),
      core.buckyball
    )
  }

  def ant(index: Int): AntCoreParams = {
    val core = cores(index) match {
      case value: AntTileCore => value
      case _ => throw new IllegalArgumentException("Ant parameters require an Ant slot")
    }
    AntCoreParams(core.buckyball, core.local, tracking, memory.chi, signatures(index), cpus.head)
  }

}

@instantiable
abstract class TileEndpoint(p: TileLinkParams, main: Boolean) extends Module with HasTLink {
  require(p.main == main, "Software tile role must match the designed TLink role")
  @public val tlink = IO(new TLinkIO(p.main, p.cpus, p.executions, p.controls, p.bus, p.axi, p.sharedStorage))
}
