package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig


// ================================================================================
// BBus Spec:
//
// 前端输入： RS: cmd, MemDomain(bridge): n0 * sramRead(), m0 * sramWrite()
// 后端输出： Ball: cmd, n1 * sramRead(), m1 * sramWrite()
// 
// 要求: n0 >= n1, m0 >= m1
// ================================================================================

/** BBus模块 - diplomacy的sink端，接收多个Ball的连接和系统输入 */
// class BBus[T <: Data, U <: Data](params: BallParams, domaincmd: T, domainbridge: U)(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
class BBus(params: BallParams)(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  // sink节点，接收来自Blink的连接
  val node = new BBusNode(params)
  
  lazy val module = new LazyModuleImp(this) {
    // 协商后的边参数
    val edgeParams = node.edges.in.head
    
    val io = IO (new Bundle {
      // val cmd = Decoupled(domaincmd)
      // val bridge = Decoupled(domainbridge)
      val blink = Flipped(new BlinkBundle(edgeParams))
    })

// ================================================================================
// 连接前端
// ================================================================================






    
// ================================================================================
// 连接后端
// ================================================================================
    node.in.head._1 <> io.blink
    
    // io.blink.cmd <> io.cmd

    // BBus接口默认处理（由外部连接覆盖）
    io.blink.cmd.req.valid := false.B
    io.blink.cmd.req.bits := DontCare
    io.blink.cmd.resp.ready := false.B
    
    io.blink.status.ready := true.B
    
    // SRAM接口默认处理
    for (i <- 0 until edgeParams.sramReadBW) {
      io.blink.data.sramRead(i).req.ready := true.B
      io.blink.data.sramRead(i).resp.valid := io.blink.data.sramRead(i).req.valid
      io.blink.data.sramRead(i).resp.bits := DontCare
    }
    
    for (i <- 0 until edgeParams.sramWriteBW) {
      io.blink.data.sramWrite(i).req.ready := true.B
    }
  }

  override lazy val desiredName = "BBus"
}
