package sims.p2e

import org.chipsalliance.cde.config.Config

class BuckyballPolyP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 5) ++
        new examples.poly.BuckyballPolyConfig
    )
