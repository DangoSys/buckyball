package examples.toy

import org.chipsalliance.cde.config.Config
import examples.toy.tiles.WithNToyTiles
import framework.core.bbtile.{L2CacheParams, WithBBTile}

import freechips.rocketchip.subsystem.{InCluster, InSubsystem}
import freechips.rocketchip.devices.tilelink.{BootROMLocated, BootROMParams}
import constellation.channel._
import constellation.routing._
import constellation.router._
import constellation.topology._
import constellation.noc._
import scala.collection.immutable.ListMap

/** Single BBTile: 1 Rocket core + 1 Buckyball accelerator */
class BuckyballToyConfig
    extends Config(
      new WithNToyTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/** Single Rocket core only (no Buckyball) */
class RocketOnlyConfig
    extends Config(
      new WithNToyTiles(withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

// Increase BootROM size for large core counts
class WithLargeBootROM(address: BigInt = 0x80000, size: Int = 0x80000)
    extends Config((site, here, up) => {
      case BootROMLocated(InSubsystem) =>
        up(BootROMLocated(InSubsystem)).map(_.copy(address = address, size = size))
    })

// ==============================================================================
// Per-Tile Private L2 Cache Configurations
// ==============================================================================

/**
 * Helper config fragment to add per-tile L2 to all tiles.
 *
 * Usage: new WithPerTileL2(ways = 4, capacityKB = 256) ++ <your config>
 *
 * @param ways Number of ways (associativity)
 * @param capacityKB L2 cache size in KB per tile
 * @param memCycles Memory round-trip latency in cycles
 */
class WithPerTileL2(
  ways:       Int = 4,
  capacityKB: Int = 256,
  memCycles:  Int = 10)
    extends Config((site, here, up) => {
      case framework.core.bbtile.BBTileAttachParams(tileParams, crossing) =>
        val cacheBlockBytes = site(freechips.rocketchip.subsystem.CacheBlockBytes)
        val sets            = (capacityKB * 1024) / (cacheBlockBytes * ways)
        framework.core.bbtile.BBTileAttachParams(
          tileParams.copy(
            l2cache = Some(L2CacheParams(
              ways = ways,
              sets = sets,
              writeBytes = 8,
              portFactor = 2,
              memCycles = memCycles
            ))
          ),
          crossing
        )
    })

/**
 * Toy configuration with per-tile private L2 cache.
 *
 * Each BBTile gets a 256KB private L2 (4-way, 256 sets, 64B blocks).
 * System-level L3 (InclusiveCache) is still present for coherence.
 */
class BuckyballToyConfigWithL2
    extends Config(
      new WithPerTileL2(ways = 4, capacityKB = 256) ++
        new WithNToyTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/**
 * Multi-tile configuration with per-tile L2.
 *
 * Each tile has 512KB private L2, plus shared system-level L3.
 * Suitable for compute-intensive workloads with good locality.
 */
class BuckyballToyMultiTileWithL2
    extends Config(
      new WithPerTileL2(ways = 8, capacityKB = 512, memCycles = 10) ++
        new WithNToyTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
