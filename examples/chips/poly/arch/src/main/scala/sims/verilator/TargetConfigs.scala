package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballPolyVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 20) ++
        new WithCustomBootROM ++
        new examples.poly.BuckyballPolyConfig
    )
