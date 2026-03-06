package framework.core.bbtile

import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, RocketCoreParams}
import freechips.rocketchip.tile.{InstantiableTileParams, RocketTileBoundaryBufferParams}
import freechips.rocketchip.subsystem.HierarchicalElementCrossingParamsLike
import freechips.rocketchip.prci.ClockSinkParameters
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.LookupByHartIdImpl
import framework.top.GlobalConfig

/**
 * Parameters for a BBTile.
 *
 * A BBTile contains N Rocket cores, each optionally paired with a Buckyball accelerator.
 * Internal modules use @instantiable + config style; diplomacy is only used for TileLink ports.
 */
case class BBTileParams(
  nCores:          Int = 1,
  withBuckyball:   Boolean = true,
  core:            RocketCoreParams = RocketCoreParams(),
  icache:          Option[ICacheParams] = Some(ICacheParams()),
  dcache:          Option[DCacheParams] = Some(DCacheParams()),
  btb:             Option[BTBParams] = Some(BTBParams()),
  buckyballConfig: GlobalConfig = GlobalConfig(),
  tileId:          Int = 0,
  beuAddr:         Option[BigInt] = None,
  blockerCtrlAddr: Option[BigInt] = None,
  clockSinkParams: ClockSinkParameters = ClockSinkParameters(),
  boundaryBuffers: Option[RocketTileBoundaryBufferParams] = None)
    extends InstantiableTileParams[BBTile] {
  require(icache.isDefined)
  require(dcache.isDefined)
  require(nCores >= 1)

  val baseName   = "bbtile"
  val uniqueName = s"${baseName}_$tileId"

  def instantiate(
    crossing:   HierarchicalElementCrossingParamsLike,
    lookup:     LookupByHartIdImpl
  )(
    implicit p: Parameters
  ): BBTile =
    new BBTile(this, crossing, lookup)

}
