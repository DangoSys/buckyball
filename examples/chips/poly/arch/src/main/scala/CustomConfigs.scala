package examples.poly

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

/** Four Poly tiles: 3 prefill + 2 decode Cores per tile (20 harts). */
class BuckyballPolyConfig
    extends Config(
      new Config((site, here, up) => { case MaxHartIdBits =>
        log2Ceil(20)
      }) ++
        new WithBuckyballTiles("../examples/chips/poly/generated/chip.pb") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
