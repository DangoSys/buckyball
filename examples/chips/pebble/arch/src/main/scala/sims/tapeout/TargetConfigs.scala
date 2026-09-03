package sims.tapeout

import org.chipsalliance.cde.config.Config
import framework.system.tile.WithBuckyballTiles

class BuckyballPebbleTapeoutConfig
    extends Config(
      new freechips.rocketchip.subsystem.WithoutTLMonitors ++
        new testchipip.soc.WithNoScratchpads ++
        new chipyard.WithTapeoutSingleClock(100) ++
        new chipyard.harness.WithSimTSIOverSerialTL(fast = true) ++
        new chipyard.harness.WithSimI2CEepromOnPads ++
        new chipyard.WithSerialConnect ++
        new chipyard.iobinders.WithSPIIOCells ++
        new chipyard.iobinders.WithSimI2CIOCells ++
        new chipyard.config.WithUART(
          baudrate = 115200,
          address = 0x10020000,
          txEntries = 8,
          rxEntries = 8
        ) ++
        new chipyard.config.WithNoUART ++
        new chipyard.config.WithSPI(address = 0x10031000) ++
        new chipyard.config.WithI2C(address = 0x10040000) ++
        new chipyard.config.WithGPIO(address = 0x10010000, width = 8) ++
        new freechips.rocketchip.subsystem.WithInclusiveCacheDirReg(true) ++
        new freechips.rocketchip.subsystem.WithInclusiveCacheSchedulerBypass(false) ++
        new freechips.rocketchip.subsystem.WithInclusiveCache(nWays = 8, capacityKB = 16) ++
        new chipyard.WithTapeoutBootROM ++
        new WithBuckyballTiles("../examples/chips/pebble/configs/generated/chip.pb") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
