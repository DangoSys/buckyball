package framework.core.bbtile

import freechips.rocketchip.rocket.{BTBParams, DCacheParams, ICacheParams, RocketCoreParams}
import freechips.rocketchip.tile.{InstantiableTileParams, RocketTileBoundaryBufferParams}
import freechips.rocketchip.subsystem.HierarchicalElementCrossingParamsLike
import freechips.rocketchip.prci.ClockSinkParameters
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.LookupByHartIdImpl
import framework.top.GlobalConfig
import sifive.blocks.inclusivecache.InclusiveCachePortParameters

/**
 * Parameters for per-tile private L2 cache.
 *
 * @param ways Number of ways (associativity)
 * @param sets Number of sets
 * @param writeBytes Backing store update granularity
 * @param portFactor numSubBanks = (widest TL port * portFactor) / writeBytes
 * @param memCycles Number of L2 clock cycles for a memory round-trip
 * @param bufInnerInterior Buffer parameters for inner interior (towards cores, inside scheduler)
 * @param bufInnerExterior Buffer parameters for inner exterior (towards cores, outside scheduler)
 * @param bufOuterInterior Buffer parameters for outer interior (towards memory, inside scheduler)
 * @param bufOuterExterior Buffer parameters for outer exterior (towards memory, outside scheduler)
 */
case class L2CacheParams(
  ways:             Int = 4,
  sets:             Int = 256,
  writeBytes:       Int = 8,
  portFactor:       Int = 2,
  memCycles:        Int = 10,
  bufInnerInterior: InclusiveCachePortParameters = InclusiveCachePortParameters.fullC,
  bufInnerExterior: InclusiveCachePortParameters = InclusiveCachePortParameters.flowAD,
  bufOuterInterior: InclusiveCachePortParameters = InclusiveCachePortParameters.full,
  bufOuterExterior: InclusiveCachePortParameters = InclusiveCachePortParameters.none)

/**
 * Parameters for a BBTile.
 *
 * A BBTile contains one Rocket core and N buckyball slots.
 * Each slot can be enabled/disabled independently, with independent GlobalConfig.
 * Internal modules use @instantiable + config style; diplomacy is only used for TileLink ports.
 *
 * @param l2cache Optional per-tile private L2 cache. When None, tile connects directly to system bus.
 *                When Some, tile traffic routes through private L2 before reaching system bus.
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
  l2cache:          Option[L2CacheParams] = None,
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
