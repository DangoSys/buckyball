package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballToyVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.toy.BuckyballToyConfig
    )

class BuckyballToyRushBVerilatorConfig
    extends Config(
      new WithBuckyballRushB ++
        new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.toy.BuckyballToyConfig
    )

class BuckyballToy8CoreVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 8) ++
        new WithCustomBootROM ++
        new examples.toy.BuckyballToy8CoreConfig
    )
