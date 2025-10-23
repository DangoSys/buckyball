package examples.gemmini

import chisel3._
import org.chipsalliance.cde.config._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.tile._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.CustomBuckyBallConfig
import framework.rocket.BuildRoCCBB

/**
 * Gemmini配置
 */
class BuckyBallGemminiConfig(
  buckyballConfig: CustomBuckyBallConfig = CustomBuckyBallConfig()
) extends Config((site, here, up) => {
  case BuildRoCCBB => up(BuildRoCCBB) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val gemmini = LazyModule(new GemminiBuckyBall(buckyballConfig))
      gemmini
    }
  )
})

/**
 * 完整的Gemmini系统配置
 */
class BuckyBallGemminiSystemConfig extends Config(
  new BuckyBallGemminiConfig ++
  new framework.rocket.WithNBuckyBallCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
