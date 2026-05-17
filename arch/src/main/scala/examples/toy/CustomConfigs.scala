package examples.toy

import org.chipsalliance.cde.config.Config
import framework.system.tile.WithBuckyballTiles

/**
 * Toy example: 1 BBTile × 1 Buckyball core.
 *
 * Demonstrates the simplest possible Buckyball configuration — a single tile
 * with a single accelerator-bearing core. All topology is defined in toy.toml
 * and its included sub-files (tiles/, cores/, balldomains/).
 */
class BuckyballToyConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/toy/configs/toy.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/**
 * Rocket-only variant: same topology as BuckyballToyConfig but with every
 * Buckyball slot torn down (Rocket cores only, no accelerators).
 */
class RocketOnlyToyConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/toy/configs/toy.toml", withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
