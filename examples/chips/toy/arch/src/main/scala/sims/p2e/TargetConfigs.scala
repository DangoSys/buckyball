package sims.p2e

import org.chipsalliance.cde.config.Config

class P2EToyConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.toy.BuckyballToyConfig
    )

class P2EToy8CoreConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 8) ++
        new examples.toy.BuckyballToy8CoreConfig
    )

/**
 * Linux variant of P2EToyConfig.
 * Uses bootrom/linux/bootrom.rv64.img which jumps to OpenSBI fw_payload at 0x80000000.
 * Pair with OpenSBI fw_payload built by `bbdev kernel --build`.
 */
class P2EToyLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new examples.toy.BuckyballToyConfig
    )

class P2EToy4CoreLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 4) ++
        new examples.toy.BuckyballToy4CoreConfig
    )

class P2EToy8CoreLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 8) ++
        new examples.toy.BuckyballToy8CoreConfig
    )

class P2EToy16CoreLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 16) ++
        new examples.toy.BuckyballToy16CoreConfig
    )
