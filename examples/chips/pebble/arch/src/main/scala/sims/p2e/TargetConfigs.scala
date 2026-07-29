package sims.p2e

import org.chipsalliance.cde.config.Config

class P2EPebbleConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.pebble.BuckyballPebbleConfig
    )

class P2EPebbleLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new examples.pebble.BuckyballPebbleConfig
    )
