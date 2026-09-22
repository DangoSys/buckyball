package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballPolyVerilatorConfig
    extends Config(
      new freechips.rocketchip.subsystem.WithoutTLMonitors ++
        new BBSimConfig(maxHarts = 20) ++
        new WithCustomBootROM ++
        new examples.poly.BuckyballPolyConfig
    )
