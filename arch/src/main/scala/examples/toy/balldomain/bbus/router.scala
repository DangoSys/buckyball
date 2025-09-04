package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{BBusNode, BallParams, BlinkBundle}

class BBusRouter(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  val node = new BBusNode(BallParams(sramReadBW = b.sp_banks, sramWriteBW = b.sp_banks)) 

  lazy val module = new LazyModuleImp(this) {
    val io = IO(new Bundle {
      val blink = Flipped(new BlinkBundle(node.edges.in.head))
    })
  }
}
