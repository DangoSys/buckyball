package examples.spark

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

/** Four Spark tiles: 2 Attention + 3 FFN Cores per tile (20 harts). */
class BuckyballSparkConfig
    extends Config(
        new WithBuckyballTiles("../examples/chips/spark/configs/generated/chip.pb") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
