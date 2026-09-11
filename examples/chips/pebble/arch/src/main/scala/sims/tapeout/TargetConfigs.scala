package sims.tapeout

import org.chipsalliance.cde.config.Config
import framework.system.tile.WithBuckyballTiles

class BuckyballPebbleTapeoutConfig
    extends Config(
      new freechips.rocketchip.subsystem.WithInclusiveCacheDirReg(true) ++
        new freechips.rocketchip.subsystem.WithInclusiveCacheSchedulerBypass(false) ++
        new freechips.rocketchip.subsystem.WithInclusiveCache(nWays = 8, capacityKB = 16) ++
        new chipyard.WithTapeoutBootROM ++
        new WithBuckyballTiles("../examples/chips/pebble/configs/generated/chip.pb") ++
        new chipyard.WithTapeoutPlatform
    )
