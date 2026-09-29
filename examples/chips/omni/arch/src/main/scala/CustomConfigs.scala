package examples.omni

import chisel3.util.log2Ceil
import org.chipsalliance.cde.config.Config
import freechips.rocketchip.tile.MaxHartIdBits
import framework.system.tile.WithBuckyballTiles

class BuckyballOmniConfig
    extends Config(
      new Config((site, here, up) => { case MaxHartIdBits => log2Ceil(5) }) ++
        new WithBuckyballTiles("../examples/chips/omni/configs/generated/chip.pb") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
