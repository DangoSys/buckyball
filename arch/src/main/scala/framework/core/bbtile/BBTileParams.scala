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
 * A BBTile contains one Rocket core and N buckyball slots.
 * Each slot can be enabled/disabled independently, with independent GlobalConfig.
 * Internal modules use @instantiable + config style; diplomacy is only used for TileLink ports.
 */
case class BBTileParams(
  nCores:           Int = 1,
  withBuckyball:    Boolean = true,
  core:             RocketCoreParams = RocketCoreParams(),
  icache:           Option[ICacheParams] = Some(ICacheParams()),
  dcache:           Option[DCacheParams] = Some(DCacheParams()),
  btb:              Option[BTBParams] = Some(BTBParams()),
  buckyballConfig:  GlobalConfig = GlobalConfig(),
  buckyballPerCore: Seq[Option[GlobalConfig]] = Nil,
  tileId:           Int = 0,
  beuAddr:          Option[BigInt] = None,
  blockerCtrlAddr:  Option[BigInt] = None,
  clockSinkParams:  ClockSinkParameters = ClockSinkParameters(),
  boundaryBuffers:  Option[RocketTileBoundaryBufferParams] = None)
    extends InstantiableTileParams[BBTile] {
  require(icache.isDefined)
  require(dcache.isDefined)
  require(nCores >= 1)
  require(
    buckyballPerCore.isEmpty || buckyballPerCore.size == nCores,
    s"buckyballPerCore size (${buckyballPerCore.size}) must be 0 or nCores ($nCores)"
  )

  val baseName   = "bbtile"
  val uniqueName = s"${baseName}_$tileId"

  val resolvedBuckyballPerCore: Seq[Option[GlobalConfig]] =
    if (buckyballPerCore.nonEmpty) buckyballPerCore
    else Seq.fill(nCores)(if (withBuckyball) Some(buckyballConfig) else None)

  val withAnyBuckyball: Boolean = resolvedBuckyballPerCore.exists(_.isDefined)

  val enabledBuckyballCores: Seq[Int] = resolvedBuckyballPerCore.zipWithIndex.collect {
    case (Some(_), i) => i
  }

  def instantiate(
    crossing:   HierarchicalElementCrossingParamsLike,
    lookup:     LookupByHartIdImpl
  )(
    implicit p: Parameters
  ): BBTile =
    new BBTile(this, crossing, lookup)

}
