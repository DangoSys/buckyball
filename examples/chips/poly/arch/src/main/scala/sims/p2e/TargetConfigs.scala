package sims.p2e

import org.chipsalliance.cde.config.Config

class BuckyballPolyP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 20) ++
        new examples.poly.BuckyballPolyConfig
    )

class BuckyballPolyLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig(maxHarts = 20) ++
        new examples.poly.BuckyballPolyConfig
    )
