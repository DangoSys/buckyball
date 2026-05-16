package framework.system.tile

import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.{InstantiableTileParams, LookupByHartIdImpl, RocketTileBoundaryBufferParams}
import freechips.rocketchip.subsystem.HierarchicalElementCrossingParamsLike
import freechips.rocketchip.prci.ClockSinkParameters
import framework.system.core.BBCoreParams
import sifive.blocks.inclusivecache.InclusiveCachePortParameters

/**
 * L2 Cache 参数
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
 * BBTile 顶层参数
 *
 * 一个 Tile = N 个 Core + IOManager + 可选 L2 Cache
 *
 * @param cores Core 列表
 * @param ioManager IOManager 参数
 * @param l2cache 可选的 L2 Cache 参数
 * @param tileId Tile ID
 */
case class BBTileParams(
  cores:           Seq[BBCoreParams],
  ioManager:       IOManagerParams,
  l2cache:         Option[L2CacheParams] = None,
  tileId:          Int = 0,
  clockSinkParams: ClockSinkParameters = ClockSinkParameters(),
  boundaryBuffers: Option[RocketTileBoundaryBufferParams] = None,
  blockerCtrlAddr: Option[BigInt] = None)
    extends InstantiableTileParams[BBTile] {

  require(cores.nonEmpty, "Tile must have at least 1 Core")
  require(
    ioManager.nCores == cores.size,
    s"IOManager nCores (${ioManager.nCores}) must match cores.size (${cores.size})"
  )

  val baseName   = "bbtile"
  val uniqueName = s"${baseName}_$tileId"

  // === TileParams trait requirements (delegated to first Core) ===
  // BaseTile expects these from its TileParams. We delegate to the first Core's
  // SMParams.rocket / l1ICache / l1DCache / btb so that BaseTile can build its
  // outer.frontend / outer.dcache / etc. via HasICacheFrontend/HasHellaCache mixins.
  val core:   freechips.rocketchip.tile.CoreParams             = cores.head.smParams.rocket
  val icache: Option[freechips.rocketchip.rocket.ICacheParams] = Some(cores.head.l1ICache)
  val dcache: Option[freechips.rocketchip.rocket.DCacheParams] = Some(cores.head.l1DCache)
  val btb:    Option[freechips.rocketchip.rocket.BTBParams]    = cores.head.btb

  /** 总 SM 数 (所有 Core 的 SM 之和) */
  val totalSMs: Int = cores.map(_.nSMs).sum

  /** 总 hartid 数 */
  val totalHarts: Int = totalSMs

  /** 是否有任何 Buckyball */
  val hasAnyBuckyball: Boolean = cores.exists(_.smParams.hasBuckyball)

  /** 计算指定 (coreId, smId) 的 hartid */
  def hartIdOf(coreId: Int, smId: Int): Int = {
    require(coreId < cores.size, s"coreId $coreId out of range")
    require(smId < cores(coreId).nSMs, s"smId $smId out of range for core $coreId")
    val priorSMs = cores.take(coreId).map(_.nSMs).sum
    priorSMs + smId
  }

  def instantiate(
    crossing:   HierarchicalElementCrossingParamsLike,
    lookup:     LookupByHartIdImpl
  )(
    implicit p: Parameters
  ): BBTile =
    new BBTile(this, crossing, lookup)

}
