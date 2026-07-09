package examples.pebble

import org.chipsalliance.cde.config.Config
import framework.system.tile.WithBuckyballTiles

/**
 * Pebble example: one tile with TransposeBall and SystolicArrayBall.
 */
class BuckyballPebbleConfig
    extends Config(
      new WithBuckyballTiles("../examples/chips/pebble/arch/src/main/scala/configs/pebble.toml") ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )

class RocketOnlyPebbleConfig
    extends Config(
      new WithBuckyballTiles("../examples/chips/pebble/arch/src/main/scala/configs/pebble.toml", withBuckyball = false) ++
        new chipyard.config.WithSystemBusWidth(128) ++
        new sims.base.BuckyballBaseConfig
    )
