package sims.firesim

import org.chipsalliance.cde.config.Config

class FireSimBuckyballPebbleConfig
    extends Config(
      new WithBootROM ++
        new firechip.chip.WithDefaultFireSimBridges ++
        new firechip.chip.WithFireSimConfigTweaks ++
        new examples.pebble.BuckyballPebbleConfig
    )
