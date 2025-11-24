package examples

import chisel3._
import framework.builtin.BaseConfig
import examples.toy.BuckyballToyConfig

object BuckyballConfigs {
  val defaultConfig = BaseConfig
  val toyConfig = BuckyballToyConfig.defaultConfig

  // Actually used configuration
  val customConfig = toyConfig

  type CustomBuckyballConfig = BaseConfig
}


// Get currently selected configuration
object CustomBuckyballConfig {
  import BuckyballConfigs._
  def apply(): CustomBuckyballConfig = BuckyballConfigs.customConfig
}
