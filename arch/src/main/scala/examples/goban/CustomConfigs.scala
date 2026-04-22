package examples.goban

import org.chipsalliance.cde.config.Config
import examples.goban.tiles.WithNGobanTiles

/** 1 BBTile × 4 buckyball slots (shared SharedMem + BarrierUnit) */
class BuckyballGobanConfig
    extends Config(
      new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )

/** 2 BBTiles × 4 slots = 8 total slots */
class BuckyballGoban2TileConfig
    extends Config(
      new WithNGobanTiles ++
        new WithNGobanTiles ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new chipyard.config.AbstractConfig
    )
