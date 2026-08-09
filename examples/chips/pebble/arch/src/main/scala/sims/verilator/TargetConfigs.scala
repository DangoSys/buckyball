package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballPebbleVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.pebble.BuckyballPebbleConfig
    )

class BuckyballPebbleHostRushVerilatorConfig
    extends Config(
      new WithBuckyballHostRush ++
        new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.pebble.BuckyballPebbleConfig
    )
