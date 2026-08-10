package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballPolyVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 5) ++
        new WithCustomBootROM ++
        new examples.poly.BuckyballPolyConfig
    )
