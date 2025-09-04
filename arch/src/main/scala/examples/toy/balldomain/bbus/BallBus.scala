package examples.toy.balldomain.bbus

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.diplomacy._
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.blink.{BBusNode, BallParams}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}

class BallBus(maxReadBW: Int, maxWriteBW: Int, numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends LazyModule {
  // 创建多个BBusNode
  val ballNodes = Seq.fill(numBalls) { new BBusNode(BallParams(sramReadBW = maxReadBW, sramWriteBW = maxWriteBW)) }

  lazy val module = new LazyModuleImp(this) {
    // 创建外部SRAM接口
    val io = IO(new Bundle {
      val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, b.spad_w)))
      val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, b.spad_w, b.spad_mask_len)))
      val accRead = Vec(b.acc_banks, Flipped(new SramReadIO(b.acc_bank_entries, b.acc_w)))
      val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))
    })
    
    // 连接外部SRAM到各个Ball节点
    // 这里实现SRAM资源的仲裁和路由
    ballNodes.foreach { node =>
      // 前b.sp_banks个端口连接到scratchpad
      for (i <- 0 until b.sp_banks) {
        node.in.head._1.data.sramRead(i) <> io.sramRead(i)
        node.in.head._1.data.sramWrite(i) <> io.sramWrite(i)
      }
      
      // 接下来b.acc_banks个端口连接到accumulator
      for (i <- 0 until b.acc_banks) {
        val readIdx = b.sp_banks + i
        val writeIdx = b.sp_banks + i
        if (readIdx < maxReadBW) {
          node.in.head._1.data.sramRead(readIdx) <> io.accRead(i)
        }
        if (writeIdx < maxWriteBW) {
          node.in.head._1.data.sramWrite(writeIdx) <> io.accWrite(i)
        }
      }
      
      // 处理多余的端口
      for (i <- (b.sp_banks + b.acc_banks) until maxReadBW) {
        node.in.head._1.data.sramRead(i).req.valid := false.B
        node.in.head._1.data.sramRead(i).req.bits := DontCare
        node.in.head._1.data.sramRead(i).resp.ready := false.B
      }
      for (i <- (b.sp_banks + b.acc_banks) until maxWriteBW) {
        node.in.head._1.data.sramWrite(i).req.valid := false.B
        node.in.head._1.data.sramWrite(i).req.bits := DontCare
      }
      
      // 设置控制接口
      node.in.head._1.cmd.req.ready := true.B
      node.in.head._1.cmd.resp.valid := false.B
      node.in.head._1.cmd.resp.bits := DontCare
      node.in.head._1.status.ready := true.B
    }
  }

  override lazy val desiredName = "BallBus"
}