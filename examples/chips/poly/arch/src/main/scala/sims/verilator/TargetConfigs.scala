package sims.verilator

import org.chipsalliance.cde.config.{Config, Parameters}

class BuckyballPolyVerilatorConfig
    extends Config(
      (if (sys.env.get("CI").contains("true"))
         new freechips.rocketchip.subsystem.WithoutTLMonitors
       else
         new Config(Parameters.empty)) ++
        new BBSimConfig(maxHarts = 20) ++
        new WithCustomBootROM ++
        new examples.poly.BuckyballPolyConfig
    )
