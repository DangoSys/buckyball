package sims.p2e

import org.chipsalliance.cde.config.Config

class BuckyballSparkP2EConfig
    extends Config(
      new P2EBaseConfig ++
        new examples.spark.BuckyballSparkConfig
    )

class BuckyballSparkLinuxP2EConfig
    extends Config(
      new WithLinuxBootROM ++
        new P2EBaseConfig ++
        new examples.spark.BuckyballSparkConfig
    )
