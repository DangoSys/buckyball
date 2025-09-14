package examples.toy.balldomain.im2col

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{BallNode, BBusParams}
import prototype.im2col.Im2col

class Im2colBall(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  // 创建自己的node，像AdderDriver一样
  val node = new BallNode(Seq(BBusParams(sramReadBW = b.sp_banks, sramWriteBW = b.sp_banks)))

  lazy val module = new LazyModuleImp(this) {
    // 实例化Im2col核心模块
    val im2col = Module(new Im2col)

    // 创建外部IO
    val io = IO(new Bundle {
      val cmdReq = Flipped(Decoupled(new examples.toy.balldomain.rs.BallRsIssue))
      val cmdResp = Decoupled(new examples.toy.balldomain.rs.BallRsComplete)
    })

    // 连接命令接口
    im2col.io.cmdReq <> io.cmdReq
    im2col.io.cmdResp <> io.cmdResp

    // 检查协商后的参数
    val negotiatedParams = node.edges.out.map(e => (e.sramReadBW, e.sramWriteBW))
    require(negotiatedParams.forall(p => p._1 >= b.sp_banks && p._2 >= b.sp_banks),
            "negotiated bandwidth must support Im2col requirements")

    // 通过diplomacy连接 - 类似AdderDriver的node.out.foreach
    node.out.foreach { case (bundle, edge) =>
      // Im2col只需要前b.sp_banks个SRAM端口
      for (i <- 0 until b.sp_banks) {
        bundle.data.sramRead(i) <> im2col.io.sramRead(i)
        bundle.data.sramWrite(i) <> im2col.io.sramWrite(i)
      }

      // 处理多余的端口
      for (i <- b.sp_banks until edge.sramReadBW) {
        bundle.data.sramRead(i).req.ready := false.B
        bundle.data.sramRead(i).resp.valid := false.B
        bundle.data.sramRead(i).resp.bits := DontCare
      }
      for (i <- b.sp_banks until edge.sramWriteBW) {
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

  override lazy val desiredName = "Im2colBall"
}
