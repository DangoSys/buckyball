package examples.poly

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

class WithPolyHartIdBits extends Config((site, here, up) => {
  case MaxHartIdBits => log2Ceil(5)
})

/** One Poly Tile: 3 prefill Cores followed by 2 decode Cores. */
class BuckyballPolyConfig
    extends Config(
      new WithPolyHartIdBits ++
        new WithBuckyballTiles("../examples/chips/poly/configs/poly.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
