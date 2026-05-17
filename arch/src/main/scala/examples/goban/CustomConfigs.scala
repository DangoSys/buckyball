package examples.goban

import org.chipsalliance.cde.config.Config
import framework.system.tile.WithBuckyballTiles

/** 1 BBTile × 2 Buckyball cores. */
class BuckyballGoban2CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/1t2c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 4 Buckyball cores (default). */
class BuckyballGoban4CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/1t4c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 8 Buckyball cores. */
class BuckyballGoban8CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/1t8c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 16 Buckyball cores. */
class BuckyballGoban16CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/1t16c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 24 BBTiles × 16 Buckyball cores each = 384 total. */
class BuckyballGoban24Tile16CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/24t16c.toml") ++
        new chipyard.config.WithSystemBusWidth(256) ++
        new sims.base.BuckyballBaseConfig
    )

/** Rocket-only variant of the default goban topology (no Buckyball). */
class RocketOnlyGobanConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/goban.toml", withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
