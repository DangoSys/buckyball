package sims.p2e

import org.chipsalliance.cde.config.Config

class BuckyballGoban2CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban2CoreConfig
    )

class BuckyballGoban4CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban4CoreConfig
    )

class BuckyballGoban8CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban8CoreConfig
    )

class BuckyballGoban16CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban16CoreConfig
    )

class BuckyballGoban64CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban64CoreConfig
    )

class BuckyballGoban24Tile16CoreP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.goban.BuckyballGoban24Tile16CoreConfig
    )

class BuckyballGoban2Tile4CoreP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 8) ++
        new examples.goban.BuckyballGoban2Tile4CoreConfig
    )

/**
 * Linux variant of the 2-tile Goban config.
 * Uses bootrom/linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 */
class BuckyballGoban2Tile4CoreLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 8) ++
        new examples.goban.BuckyballGoban2Tile4CoreConfig
    )

class BuckyballGoban64Tile4CoreP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 256) ++
        new examples.goban.BuckyballGoban64Tile4CoreConfig
    )

/**
 * Linux variant of the 64-tile Goban config.
 * Uses bootrom/linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 */
class BuckyballGoban64Tile4CoreLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 256) ++
        new examples.goban.BuckyballGoban64Tile4CoreConfig
    )

class BuckyballGobanConfig1LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 60) ++
        new examples.goban.BuckyballGobanConfig1Config
    )

class BuckyballGobanConfig2LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 64) ++
        new examples.goban.BuckyballGobanConfig2Config
    )

class BuckyballGobanConfig3LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 64) ++
        new examples.goban.BuckyballGobanConfig3Config
    )

class BuckyballGobanConfig4LinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 64) ++
        new examples.goban.BuckyballGobanConfig4Config
    )
