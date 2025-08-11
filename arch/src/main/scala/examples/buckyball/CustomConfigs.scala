package buckyball

import org.chipsalliance.cde.config.{Config, Parameters, Field}
import chisel3._
import freechips.rocketchip.diplomacy.LazyModule
import freechips.rocketchip.subsystem.SystemBusKey
// import freechips.rocketchip.tile.{BuildRoCC, OpcodeSet}
import framework.ballcore.ballcore._
import framework.ballcore._



object BuckyBallCustomConfigs {
  val defaultConfig = BuckyBallConfigs.defaultConfig
  val customConfig = defaultConfig
}

class BuckyBallCustomConfig(
  buckyballConfig: BuckyBallConfig = BuckyBallCustomConfigs.customConfig
) extends Config((site, here, up) => {
  case BuildRoCCBB => up(BuildRoCCBB) ++ Seq(
    (p: Parameters) => {
      implicit val q = p
      val buckyball = LazyModule(new BuckyBall(buckyballConfig))
      buckyball
    }
  )
})

class BuckyBallRocketConfig extends Config(
  new buckyball.BuckyBallCustomConfig ++
  new freechips.rocketchip.rocket.WithNHugeCores(1) ++
  new chipyard.config.WithSystemBusWidth(128) ++
  new chipyard.config.AbstractConfig
)
