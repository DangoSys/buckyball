package sims.p2e

import org.chipsalliance.cde.config.Config

class MultiRocket32CoreP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 32) ++
        new examples.multirocket.MultiRocket32CoreConfig
    )

class MultiRocket48CoreP2EConfig
    extends Config(
      new P2EBaseConfig(maxHarts = 48) ++
        new examples.multirocket.MultiRocket48CoreConfig
    )
