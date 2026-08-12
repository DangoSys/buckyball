package sims.verilator

import org.chipsalliance.cde.config.Config

class MultiRocket32CoreVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 32) ++
        new WithCustomBootROM ++
        new examples.multirocket.MultiRocket32CoreConfig
    )

class MultiRocket48CoreVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 48) ++
        new WithCustomBootROM ++
        new examples.multirocket.MultiRocket48CoreConfig
    )
