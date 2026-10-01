package sims.p2e

import org.chipsalliance.cde.config.Config

class BuckyballOmniLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 5) ++
        new examples.omni.BuckyballOmniConfig
    )
