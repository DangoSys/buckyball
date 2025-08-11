import chisel3._
// _root_ disambiguates from package chisel3.util.circt if user imports chisel3.util._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters


import buckyball.BuckyBall
import buckyball.BuckyBallConfigs

object Elaborate extends App {
  // Create a BuckyBall instance with default configuration
  val buckyball = new BuckyBall(BuckyBallConfigs.defaultConfig)(Parameters.empty)
  
  ChiselStage.emitSystemVerilogFile(
    buckyball.module,
    firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info", "-default-layer-specialization=enable")
  )
}       