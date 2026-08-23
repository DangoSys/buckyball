package examples.multirocket

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

class WithMultiRocketHartIdBits(nTiles: Int, nCoresPerTile: Int)
    extends Config((site, here, up) => { case MaxHartIdBits =>
      log2Ceil(nTiles * nCoresPerTile)
    })

/** 1 tile × 32 Rocket-only cores (no Buckyball). */
class MultiRocket32CoreConfig
    extends Config(
      new WithMultiRocketHartIdBits(nTiles = 1, nCoresPerTile = 32) ++
        new WithBuckyballTiles(
          "../examples/chips/multi-rocket/generated/chip.pb"
        ) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

/** 1 tile × 48 Rocket-only cores (no Buckyball). */
class MultiRocket48CoreConfig
    extends Config(
      new WithMultiRocketHartIdBits(nTiles = 1, nCoresPerTile = 48) ++
        new WithBuckyballTiles(
          "../examples/chips/multi-rocket/generated/1t48c.pb"
        ) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
