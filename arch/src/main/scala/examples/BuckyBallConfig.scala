package examples

import chisel3._
import framework.builtin.BaseConfig
import examples.toy.BuckyBallToyConfig

object BuckyBallConfigs {
  val defaultConfig = BaseConfig
  val toyConfig = BuckyBallToyConfig.defaultConfig

  // 实际使用的配置
  val customConfig = toyConfig

  type CustomBuckyBallConfig = BaseConfig
}


// 获取当前选择的配置
object CustomBuckyBallConfig {
  import BuckyBallConfigs._
  def apply(): CustomBuckyBallConfig = BuckyBallConfigs.customConfig
}
