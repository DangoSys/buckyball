package examples.goban

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

class WithGobanHiddenHartIdBits(nTiles: Int, nCoresPerTile: Int, hiddenHartBase: Int)
    extends Config((site, here, up) => {
      case MaxHartIdBits => log2Ceil(hiddenHartBase + nTiles * (nCoresPerTile - 1))
    })

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

/** 1 BBTile × 32 Buckyball cores. */
class BuckyballGoban32CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/1t32c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 BBTile × 64 Buckyball cores. */
class BuckyballGoban64CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/1t64c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 4 BBTile × 8 Buckyball cores. */
// this config is incoherent
class BuckyballGoban4Tile8CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/4t16c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 4 BBTile × 16 Buckyball cores. */
// this config is incoherent
class BuckyballGoban4Tile16CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/4t16c.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 8 BBTile × 8 Buckyball cores. */
// this config is incoherent
class BuckyballGoban8Tile8CoreConfig
    extends Config(
      new WithBuckyballTiles("src/main/scala/examples/goban/configs/8t8c.toml") ++
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

/** 2 BBTiles × 4 Rocket cores each = 8 harts; core 0 in each tile has Buckyball. */
class BuckyballGoban2Tile4CoreConfig
    extends Config(
      new WithGobanHiddenHartIdBits(nTiles = 2, nCoresPerTile = 4, hiddenHartBase = 2) ++
        new WithBuckyballTiles("src/main/scala/examples/goban/configs/2t4c-private.toml", hiddenHartBase = Some(2)) ++
        new chipyard.config.WithSystemBusWidth(256) ++
        new sims.base.BuckyballBaseConfig
    )

/** 64 BBTiles × 4 Rocket cores each = 256 harts; core 0 in each tile has Buckyball. */
class BuckyballGoban64Tile4CoreConfig
    extends Config(
      new WithGobanHiddenHartIdBits(nTiles = 64, nCoresPerTile = 4, hiddenHartBase = 64) ++
        new WithBuckyballTiles("src/main/scala/examples/goban/configs/64t4c-private.toml", hiddenHartBase = Some(64)) ++
        new chipyard.config.WithSystemBusWidth(256) ++
        new sims.base.BuckyballBaseConfig
    )

/** Config1: 64 BBTiles × 2 cores each; core 0 has Buckyball, core 1 is hidden Rocket-only. */
class BuckyballGobanConfig1Config
    extends Config(
      new WithGobanHiddenHartIdBits(nTiles = 64, nCoresPerTile = 2, hiddenHartBase = 64) ++
        new WithBuckyballTiles(
          "src/main/scala/examples/goban/configs/64t2c-private-1bb.toml",
          hiddenHartBase = Some(64)
        ) ++
        new chipyard.config.WithSystemBusWidth(256) ++
        new sims.base.BuckyballBaseConfig
    )

/** Config2: 64 BBTiles × 3 cores each; core 0 has Buckyball, core 1 Rocket-only, core 2 safe core. */
class BuckyballGobanConfig2Config
    extends Config(
      new WithGobanHiddenHartIdBits(nTiles = 64, nCoresPerTile = 3, hiddenHartBase = 64) ++
        new WithBuckyballTiles(
          "src/main/scala/examples/goban/configs/64t3c-private-1bb-safe.toml",
          hiddenHartBase = Some(64)
        ) ++
        new chipyard.config.WithSystemBusWidth(256) ++
        new sims.base.BuckyballBaseConfig
    )

/** Config3: 64 BBTiles × 2 cores each; both cores have Buckyball, core 1 is hidden from Linux. */
class BuckyballGobanConfig3Config
    extends Config(
      new WithGobanHiddenHartIdBits(nTiles = 64, nCoresPerTile = 2, hiddenHartBase = 64) ++
        new WithBuckyballTiles(
          "src/main/scala/examples/goban/configs/64t2c-private-2bb.toml",
          hiddenHartBase = Some(64)
        ) ++
        new chipyard.config.WithSystemBusWidth(256) ++
        new sims.base.BuckyballBaseConfig
    )

/** Config4: 64 BBTiles × 4 hidden Rocket-only cores each. Linux still sees core 0 in every tile. */
class BuckyballGobanConfig4Config
    extends Config(
      new WithGobanHiddenHartIdBits(nTiles = 64, nCoresPerTile = 4, hiddenHartBase = 64) ++
        new WithBuckyballTiles(
          "src/main/scala/examples/goban/configs/64t4c-private-rocket-only.toml",
          hiddenHartBase = Some(64)
        ) ++
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
