package examples.toy

import org.chipsalliance.cde.config.{Config, Parameters, Field}
import chisel3._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey
// import freechips.rocketchip.tile.{BuildRoCC, OpcodeSet}
import freechips.rocketchip.tile._
import examples.toy.ToyBuckyBall
import framework.builtin.BaseConfig
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.CustomBuckyBallConfig

// Use standard BuildRoCC instead of BuildRoCCBB
// import framework.rocket.BuildRoCCBB
// import framework.rocket.MultiRoCCKeyBB



object BuckyBallToyConfig {
  val defaultConfig = new BaseConfig(
    inputType = UInt(8.W),
    accType = UInt(32.W),
    sp_banks = 4,
    acc_banks = 8
  )
}

class BuckyBallCustomConfig(
  buckyballConfig: CustomBuckyBallConfig = CustomBuckyBallConfig()
) extends Config((site, here, up) => {
  case BuildRoCC => up(BuildRoCC) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val buckyball = LazyModule(new ToyBuckyBall(buckyballConfig))
      buckyball
    }
  )
})

// class WithMultiRoCCToyBuckyBall(harts: Int*)(
//   buckyballConfig: CustomBuckyBallConfig = CustomBuckyBallConfig()
// ) extends Config((site, here, up) => {
//     case MultiRoCCKeyBB => up(MultiRoCCKeyBB, site) ++ harts.distinct.map { i =>
//     (i -> Seq((p: Parameters) => {
//       implicit val q = p
//       val buckyball = LazyModule(new ToyBuckyBall(buckyballConfig))
//       buckyball
//     }))
//   }
// })



class BuckyBallToyConfig extends Config(
  new BuckyBallCustomConfig ++
  new framework.rocket.WithNBuckyBallCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig)
