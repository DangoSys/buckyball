package examples.toy

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

class WithToyHartIdBits(nTiles: Int, nCoresPerTile: Int)
    extends Config((site, here, up) => {
      case MaxHartIdBits => log2Ceil(nTiles * nCoresPerTile)
    })

/**
 * Toy example: 1 BBTile × 1 Buckyball core.
 *
 * Demonstrates the simplest possible Buckyball configuration — a single tile
 * with a single accelerator-bearing core. All topology is defined in toy.toml
 * and its included topology and Core files.
 */
class BuckyballToyConfig
    extends Config(
      new WithBuckyballTiles("../examples/chips/toy/configs/toy.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 4 Buckyball cores sharing one tile-private L2. */
class BuckyballToy4CoreConfig
    extends Config(
      new WithToyHartIdBits(nTiles = 1, nCoresPerTile = 4) ++
        new WithBuckyballTiles("../examples/chips/toy/configs/1t4c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 8 Buckyball cores sharing one tile-private L2. */
class BuckyballToy8CoreConfig
    extends Config(
      new WithToyHartIdBits(nTiles = 1, nCoresPerTile = 8) ++
        new WithBuckyballTiles("../examples/chips/toy/configs/1t8c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 16 Buckyball cores sharing one tile-private L2. */
class BuckyballToy16CoreConfig
    extends Config(
      new WithToyHartIdBits(nTiles = 1, nCoresPerTile = 16) ++
        new WithBuckyballTiles("../examples/chips/toy/configs/1t16c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/**
 * Rocket-only variant: same topology as BuckyballToyConfig but with every
 * Buckyball slot torn down (Rocket cores only, no accelerators).
 */
class RocketOnlyToyConfig
    extends Config(
      new WithBuckyballTiles("../examples/chips/toy/configs/toy.toml", withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
