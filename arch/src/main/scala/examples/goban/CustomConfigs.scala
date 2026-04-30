package examples.goban

import org.chipsalliance.cde.config.Config
import examples.goban.tiles.WithNGobanTiles
import framework.core.bbtile.L2CacheParams

/** 1 BBTile × 4 buckyball slots (shared SharedMem + BarrierUnit) */
class BuckyballGobanConfig
    extends Config(
      new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/** 2 BBTiles × 4 slots = 8 total slots */
class BuckyballGoban2TileConfig
    extends Config(
      new WithNGobanTiles ++
        new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

// ==============================================================================
// Per-Tile Private L2 Cache Configurations
// ==============================================================================

/**
 * Helper config fragment to add per-tile L2 to all Goban tiles.
 *
 * Usage: new WithGobanPerTileL2(ways = 4, capacityKB = 512) ++ <your config>
 *
 * @param ways Number of ways (associativity)
 * @param capacityKB L2 cache size in KB per tile
 * @param memCycles Memory round-trip latency in cycles
 */
class WithGobanPerTileL2(
  ways:       Int = 8,
  capacityKB: Int = 512,
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
 * Goban configuration with per-tile private L2 cache.
 *
 * 1 BBTile with 512KB private L2 (8-way, 256 sets, 64B blocks).
 * Each tile has 4 Buckyball slots sharing the tile's L2.
 * System-level L3 (InclusiveCache) maintains coherence.
 */
class BuckyballGobanConfigWithL2
    extends Config(
      new WithGobanPerTileL2(ways = 8, capacityKB = 512) ++
        new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/**
 * 2-tile Goban configuration with per-tile private L2.
 *
 * 2 BBTiles, each with 512KB private L2 (8-way).
 * Total: 8 Buckyball slots (4 per tile), 1MB total L2 cache.
 * Reduces inter-tile L2 contention for multi-tile workloads.
 */
class BuckyballGoban2TileConfigWithL2
    extends Config(
      new WithGobanPerTileL2(ways = 8, capacityKB = 512) ++
        new WithNGobanTiles ++
        new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/**
 * Large L2 configuration for compute-intensive workloads.
 *
 * 2 BBTiles, each with 1MB private L2 (8-way, 512 sets).
 * Total: 2MB L2 cache for better working set coverage.
 */
class BuckyballGoban2TileConfigWithLargeL2
    extends Config(
      new WithGobanPerTileL2(ways = 8, capacityKB = 1024) ++
        new WithNGobanTiles ++
        new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
