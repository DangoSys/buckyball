package framework.top.configs

import upickle.default.{macroRW, ReadWriter}
import framework.frontend.configs.FrontendParam
import framework.memdomain.configs.MemDomainParam
import framework.rvv.configs.RvvParam
import framework.balldomain.configs.BallDomainParam
import framework.system.core.rocket.configs.RocketCpuParam
import framework.system.tile.configs.TileParam
import memcore.bus.chi
import memcore.bus.chi.rnf.RnfParams
import memcore.memory.cpu.PhysicalRegion
import memcore.memory.coherence.configs.CoherenceParams
import memcore.memory.cache.configs.CacheParams
import memcore.memory.{interlock, spm}
import memcore.memory.ddr.{Params => DdrParams}

case class AcceleratorConfig(
  memDomain:     MemDomainParam,
  frontend:      FrontendParam,
  rvv:           RvvParam,
  ballDomain:    BallDomainParam,
  tile:          TileParam,
  sim:           SimParam,
  coreSignature: BigInt)

object AcceleratorConfig { implicit val rw: ReadWriter[AcceleratorConfig] = macroRW }

case class CoreDesign(
  factory:     String,
  rocket:      Option[RocketCpuParam],
  local:       Option[framework.ant.Params],
  accelerator: Option[AcceleratorConfig]) {
  require(rocket.isDefined != local.isDefined)
}

object CoreDesign {
  import DesignCodecs._
  implicit val rw: ReadWriter[CoreDesign] = macroRW
}

case class SharedConfig(bankBits: Int, bankEntries: Int, banks: Int)
object SharedConfig { implicit val rw: ReadWriter[SharedConfig] = macroRW }

case class TileDesign(
  factory:       String,
  param:         TileParam,
  main:          Boolean,
  cores:         Seq[CoreDesign],
  signatures:    Seq[BigInt],
  controller:    Option[Int],
  sharedStorage: Option[SharedConfig])

object TileDesign { implicit val rw: ReadWriter[TileDesign] = macroRW }

case class DevicesConfig(
  clintBase:    BigInt,
  clintBytes:   BigInt,
  tickCycles:   Int,
  plicBase:     BigInt,
  plicBytes:    BigInt,
  sources:      Int,
  priorityBits: Int,
  bootrom:      Option[RomConfig],
  scuBase:      BigInt,
  scuStride:    BigInt,
  scuBytes:     BigInt,
  scuMaxHarts:  Int)

object DevicesConfig { implicit val rw: ReadWriter[DevicesConfig] = macroRW }
case class RomConfig(base: BigInt, bytes: BigInt, image: Seq[Byte])
object RomConfig     { implicit val rw: ReadWriter[RomConfig] = macroRW     }

case class PlatformConfig(
  cpuPhysicalBits: Int,
  memory:          CoherenceParams,
  l1:              RnfParams,
  regions:         Seq[PhysicalRegion],
  tracking:        interlock.Params,
  ddr:             DdrParams,
  devices:         DevicesConfig)

object PlatformConfig {
  import DesignCodecs._
  implicit val rw: ReadWriter[PlatformConfig] = macroRW
}

case class TilePlacement(design: TileDesign, hartIds: Seq[Int], executionIds: Seq[Int])
object TilePlacement { implicit val rw: ReadWriter[TilePlacement] = macroRW }
case class SystemDesign(tiles: Seq[TilePlacement], platform: PlatformConfig)
object SystemDesign  { implicit val rw: ReadWriter[SystemDesign] = macroRW  }

case class TileScope(
  design:    TileDesign,
  platform:  PlatformConfig,
  maxHartId: Int,
  controls:  Int,
  tiles:     Int)

object TileScope { implicit val rw: ReadWriter[TileScope] = macroRW }

case class CoreScope(
  factory:        String,
  rocket:         Option[RocketCpuParam],
  local:          Option[framework.ant.Params],
  hasAccelerator: Boolean,
  controller:     Option[RocketCpuParam],
  tile:           TileParam,
  platform:       PlatformConfig,
  maxHartId:      Int,
  cpuIndex:       Int,
  cpuCount:       Int,
  signature:      BigInt)

object CoreScope {
  import DesignCodecs._
  implicit val rw: ReadWriter[CoreScope] = macroRW
}

object DesignCodecs {
  implicit val chiRW:       ReadWriter[chi.Params]           = macroRW
  implicit val cacheRW:     ReadWriter[CacheParams]          = macroRW
  implicit val coherenceRW: ReadWriter[CoherenceParams]      = macroRW
  implicit val rnfRW:       ReadWriter[RnfParams]            = macroRW
  implicit val regionRW:    ReadWriter[PhysicalRegion]       = macroRW
  implicit val trackingRW:  ReadWriter[interlock.Params]     = macroRW
  implicit val ddrRW:       ReadWriter[DdrParams]            = macroRW
  implicit val spmRW:       ReadWriter[spm.Params]           = macroRW
  implicit val antRW:       ReadWriter[framework.ant.Params] = macroRW
}
