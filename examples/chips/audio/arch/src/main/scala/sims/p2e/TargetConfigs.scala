package sims.p2e

import org.chipsalliance.cde.config.Config

class P2EAudioConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.audio.BuckyballAudioConfig
    )

class P2EAudioLinuxConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new examples.audio.BuckyballAudioConfig
    )
