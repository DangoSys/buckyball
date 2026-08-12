package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballAudioVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 6) ++
        new WithCustomBootROM ++
        new examples.audio.BuckyballAudioConfig
    )
