package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballSparkVerilatorConfig
    extends Config(
      new freechips.rocketchip.subsystem.WithoutTLMonitors ++
        new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.spark.BuckyballSparkConfig
    )
