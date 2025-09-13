package uvbb

import chisel3._
import _root_.circt.stage.ChiselStage
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig

class BallTop(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    val reset = Input(Bool())
    val clock = Input(Clock())
  })

  val vecUnit = Module(new VecBallTestWrapper)

  // 连接时钟和复位
  vecUnit.clock := clock
  vecUnit.reset := reset
}

object BallTopMain extends App {
  implicit val config: CustomBuckyBallConfig = examples.CustomBuckyBallConfig()
  implicit val params: Parameters = Parameters.empty

  ChiselStage.emitSystemVerilogFile(
    new BallTop(),
    firtoolOpts = args,
    args = Array.empty
  )
}
