package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballOmniVerilatorConfig
    extends Config(
      new freechips.rocketchip.subsystem.WithoutTLMonitors ++
        new BBSimConfig(maxHarts = 5) ++
        new WithCustomBootROM ++
        new examples.omni.BuckyballOmniConfig
    )
