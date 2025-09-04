package examples.toy.balldomain.vecball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{BallNode, BBusParams}
import prototype.vector.VecUnit

class VecBall(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  // 创建自己的node，请求scratchpad + accumulator带宽
  val node = new BallNode(Seq(BBusParams(sramReadBW = b.sp_banks + b.acc_banks, sramWriteBW = b.sp_banks + b.acc_banks)))

  lazy val module = new LazyModuleImp(this) {
    // 实例化VecUnit核心模块
    val vecUnit = Module(new VecUnit)
    
    // 创建外部IO
    val io = IO(new Bundle {
      val cmdReq = Flipped(Decoupled(new examples.toy.balldomain.rs.BallRsIssue))
      val cmdResp = Decoupled(new examples.toy.balldomain.rs.BallRsComplete)
    })
    
    // 连接命令接口
    vecUnit.io.cmdReq <> io.cmdReq
    vecUnit.io.cmdResp <> io.cmdResp
    
    // 检查协商后的参数
    val negotiatedParams = node.edges.out.map(e => (e.sramReadBW, e.sramWriteBW))
    require(negotiatedParams.forall(p => p._1 >= (b.sp_banks + b.acc_banks) && p._2 >= (b.sp_banks + b.acc_banks)), 
            "negotiated bandwidth must support VecUnit requirements")
    
    // 通过diplomacy连接
    node.out.foreach { case (bundle, edge) =>
      // VecUnit需要scratchpad + accumulator
      // 前b.sp_banks个连接到scratchpad
      for (i <- 0 until b.sp_banks) {
        bundle.data.sramRead(i) <> vecUnit.io.sramRead(i)
        bundle.data.sramWrite(i) <> vecUnit.io.sramWrite(i)
      }
      
      // 接下来b.acc_banks个连接到accumulator
      for (i <- 0 until b.acc_banks) {
        val readIdx = b.sp_banks + i
        val writeIdx = b.sp_banks + i
        bundle.data.sramRead(readIdx) <> vecUnit.io.accRead(i)
        bundle.data.sramWrite(writeIdx) <> vecUnit.io.accWrite(i)
      }
      
      // 处理多余的端口
      for (i <- (b.sp_banks + b.acc_banks) until edge.sramReadBW) {
        bundle.data.sramRead(i).req.ready := false.B
        bundle.data.sramRead(i).resp.valid := false.B
        bundle.data.sramRead(i).resp.bits := DontCare
      }
      for (i <- (b.sp_banks + b.acc_banks) until edge.sramWriteBW) {
        bundle.data.sramWrite(i).req.ready := false.B
      }
      
      // 设置控制接口
      bundle.cmd.req.ready := true.B
      bundle.cmd.resp.valid := false.B
      bundle.cmd.resp.bits := DontCare
      bundle.status.valid := false.B
      bundle.status.bits := DontCare
    }
  }

  override lazy val desiredName = "VecBall"
}