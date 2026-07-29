package sims.verilator

import org.chipsalliance.cde.config.Config

class BuckyballGoban2CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban2CoreConfig
    )

class BuckyballGoban4CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban4CoreConfig
    )

class BuckyballGoban8CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban8CoreConfig
    )

class BuckyballGoban32CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban32CoreConfig
    )

class BuckyballGoban64CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban64CoreConfig
    )

class BuckyballGoban4Tile8CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban4Tile8CoreConfig
    )

class BuckyballGoban4Tile16CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban4Tile16CoreConfig
    )

class BuckyballGoban8Tile8CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban8Tile8CoreConfig
    )

class BuckyballGoban24Tile16CoreVerilatorConfig
    extends Config(
      new BBSimConfig ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban24Tile16CoreConfig
    )

class BuckyballGoban2Tile4CoreVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 8) ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban2Tile4CoreConfig
    )

class BuckyballGoban2Tile4CoreLinuxVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 8) ++
        new WithLinuxBootROM ++
        new examples.goban.BuckyballGoban2Tile4CoreConfig
    )

class BuckyballGoban64Tile4CoreVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 256) ++
        new WithCustomBootROM ++
        new examples.goban.BuckyballGoban64Tile4CoreConfig
    )

class BuckyballGoban64Tile4CoreLinuxVerilatorConfig
    extends Config(
      new BBSimConfig(maxHarts = 256) ++
        new WithLinuxBootROM ++
        new examples.goban.BuckyballGoban64Tile4CoreConfig
    )
