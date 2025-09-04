package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig

/** Ball模块 - diplomacy的source端，向上发送带宽需求 */
class Ball(params: BallParams)(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  // 创建source节点，向上发送自己的带宽需求
  val node = new BallNode(Seq(BBusParams(params.sramReadBW, params.sramWriteBW)))
  
  lazy val module = new LazyModuleImp(this) {
    // 获取协商后的边参数
    val edgeParams = node.edges.out.head
    
    // 创建与Blink连接的接口
    val io = IO(new BlinkBundle(edgeParams))
    
    // 连接到diplomacy网络
    node.out.head._1 <> io
    
    // Ball接口默认值（由各个具体Ball实现来覆盖）
    io.cmd.req.ready  := false.B
    io.cmd.resp.valid := false.B
    io.cmd.resp.bits  := DontCare
    io.status.valid   := false.B
    io.status.bits    := DontCare
    
    // SRAM接口默认值
    for (i <- 0 until edgeParams.sramReadBW) {
      io.data.sramRead(i).req.ready := false.B
      io.data.sramRead(i).resp.valid := false.B
      io.data.sramRead(i).resp.bits := DontCare
    }
    
    for (i <- 0 until edgeParams.sramWriteBW) {
      io.data.sramWrite(i).req.ready := false.B
    }
  }

  override lazy val desiredName = "Ball"
}