package examples.poly

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

/** Four Poly tiles: 2 Attention + 3 FFN Cores per tile (20 harts). */
class BuckyballPolyConfig
    extends Config(
      new Config((site, here, up) => { case MaxHartIdBits =>
        log2Ceil(20)
      }) ++
        new WithBuckyballTiles("../examples/chips/poly/configs/generated/chip.pb", useMeshSharedMem = true) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
