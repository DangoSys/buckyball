package framework.system.tile

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
case class PrivateDCacheParams(
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
 * @param privateDCache Optional per-tile private data cache (between L1 DCache and system LLC).
 *                      When None, tile connects directly to system bus.
 *                      When Some, tile traffic routes through private DCache before reaching system bus.
 * @param hiddenHartBase Optional base hart id for non-leader cores. When set,
 *                       core 0 keeps the tile id as its hart id and cores 1..N
 *                       are mapped after hiddenHartBase.
 */
case class BBTileParams(
  nCores:            Int = 1,
  withBuckyball:     Boolean = true,
  core:              RocketCoreParams = RocketCoreParams(),
  icache:            Option[ICacheParams] = Some(ICacheParams()),
  dcache:            Option[DCacheParams] = Some(DCacheParams()),
  btb:               Option[BTBParams] = Some(BTBParams()),
  buckyballConfig:   GlobalConfig = GlobalConfig(),
  buckyballPerCore:  Seq[Option[GlobalConfig]] = Nil,
  rocketCorePerCore: Seq[RocketCoreParams] = Nil,
  privateDCache:     Option[PrivateDCacheParams] = None,
  hiddenHartBase:    Option[Int] = None,
  tileId:            Int = 0,
  beuAddr:           Option[BigInt] = None,
  blockerCtrlAddr:   Option[BigInt] = None,
  clockSinkParams:   ClockSinkParameters = ClockSinkParameters(),
  boundaryBuffers:   Option[RocketTileBoundaryBufferParams] = None)
    extends InstantiableTileParams[BBTile] {
  require(icache.isDefined)
  require(dcache.isDefined)
  require(nCores >= 1)
  hiddenHartBase.foreach { base =>
    require(base > 0, s"hiddenHartBase ($base) must be positive")
  }
  require(
    buckyballPerCore.isEmpty || buckyballPerCore.size == nCores,
    s"buckyballPerCore size (${buckyballPerCore.size}) must be 0 or nCores ($nCores)"
  )
  require(
    rocketCorePerCore.isEmpty || rocketCorePerCore.size == nCores,
    s"rocketCorePerCore size (${rocketCorePerCore.size}) must be 0 or nCores ($nCores)"
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

  def rocketCoreForCore(coreIdx: Int): RocketCoreParams = {
    require(coreIdx >= 0 && coreIdx < nCores, s"coreIdx ($coreIdx) must be in [0, $nCores)")
    if (rocketCorePerCore.nonEmpty) rocketCorePerCore(coreIdx) else core
  }

  def hartIdForCore(coreIdx: Int): Int = {
    require(coreIdx >= 0 && coreIdx < nCores, s"coreIdx ($coreIdx) must be in [0, $nCores)")
    hiddenHartBase match {
      case Some(base) =>
        if (coreIdx == 0) tileId
        else base + tileId * (nCores - 1) + (coreIdx - 1)
      case None       =>
        tileId * nCores + coreIdx
    }
  }

  def instantiate(
    crossing:   HierarchicalElementCrossingParamsLike,
    lookup:     LookupByHartIdImpl
  )(
    implicit p: Parameters
  ): BBTile =
    new BBTile(this, crossing, lookup)

}
