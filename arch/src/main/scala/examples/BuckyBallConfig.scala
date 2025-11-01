package examples

import chisel3._
import framework.builtin.BaseConfig
import examples.toy.BuckyBallToyConfig

object BuckyBallConfigs {
  val defaultConfig = BaseConfig
  val toyConfig = BuckyBallToyConfig.defaultConfig

  // Actually used configuration
  val customConfig = toyConfig

  type CustomBuckyBallConfig = BaseConfig
}


// Get currently selected configuration
object CustomBuckyBallConfig {
  import BuckyBallConfigs._
  def apply(): CustomBuckyBallConfig = BuckyBallConfigs.customConfig
}
