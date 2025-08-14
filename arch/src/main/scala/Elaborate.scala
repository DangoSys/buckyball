import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters

// import chipyard.harness.TestHarness
import freechips.rocketchip.diplomacy.LazyModule
import buckyball.{BuckyBall, BuckyBallRocketConfig}
// import chipyard.ChipTop
import freechips.rocketchip.system.TestHarness

object Elaborate extends App {
  // Use the proper Chipyard configuration that includes BuckyBall
  val config = new BuckyBallRocketConfig
  val params = config.toInstance
  
  // Use ChiselStage with proper elaboration context
  ChiselStage.emitSystemVerilogFile(
    new chipyard.harness.TestHarness()(config.toInstance),
    firtoolOpts = args,
    args = Array.empty  // 直接传递命令行参数
  )
}       
