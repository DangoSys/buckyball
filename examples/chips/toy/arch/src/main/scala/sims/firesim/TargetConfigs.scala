package sims.firesim

import org.chipsalliance.cde.config.Config

class FireSimBuckyballToyConfig
    extends Config(
      new WithBootROM ++
        new firechip.chip.WithDefaultFireSimBridges ++
        new firechip.chip.WithFireSimConfigTweaks ++
        new examples.toy.BuckyballToyConfig
    )
