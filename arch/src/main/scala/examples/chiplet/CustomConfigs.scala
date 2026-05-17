package examples.chiplet

import org.chipsalliance.cde.config.Config
import framework.system.tile.WithBuckyballTiles

/** Chiplet: 1 BBTile × 1 core with private DCache enabled (default). */
class BuckyballChipletConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/chiplet/configs/chiplet.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** Chiplet: 1 BBTile × 2 cores with private DCache enabled. */
class BuckyballChiplet2CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/chiplet/configs/chiplet_2core.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** Chiplet: 1 BBTile × 4 cores with private DCache enabled. */
class BuckyballChiplet4CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/chiplet/configs/chiplet_4core.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** Chiplet: 1 BBTile × 8 cores with private DCache enabled. */
class BuckyballChiplet8CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/chiplet/configs/chiplet_8core.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** Rocket-only variant of the default chiplet topology (no Buckyball). */
class RocketOnlyChipletConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/chiplet/configs/chiplet.toml", withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
