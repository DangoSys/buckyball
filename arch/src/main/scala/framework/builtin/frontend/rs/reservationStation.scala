package framework.builtin.frontend.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain._
import framework.rocket.RoCCResponseBB
import framework.blink.BallRegist

// Ball设备信息 - 用于注册Ball设备的配置信息
case class BallRsRegist(
  ballId: Int,
  ballName: String
)

// Ball域的发射接口 - 包含全局rob_id
class BallRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd = new BallDecodeCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)  // 全局ROB ID
}

// Ball域的完成接口
class BallRsComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// 通用Ball域发射接口 - 支持动态数量的Ball设备
class BallIssueInterface(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val balls = Vec(numBalls, Decoupled(new BallRsIssue))
}

// 通用Ball域完成接口 - 支持动态数量的Ball设备
class BallCommitInterface(numBalls: Int)(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val balls = Vec(numBalls, Flipped(Decoupled(new BallRsComplete)))
}

// 局部Ball保留站 - 简单的FIFO调度器
class BallReservationStation(BallRsRegists: Seq[BallRsRegist])
  (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {

  val numBalls = BallRsRegists.length

  val io = IO(new Bundle {
    // 解码后的指令输入（带全局rob_id）
    val ball_decode_cmd_i = Flipped(new DecoupledIO(new Bundle {
      val cmd = new BallDecodeCmd
      val rob_id = UInt(log2Up(b.rob_entries).W)  // 全局ROB ID
    }))

    // Rs -> BallController (多通道发射)
    val issue_o     = new BallIssueInterface(numBalls)
    val commit_i    = new BallCommitInterface(numBalls)

    // 输出完成信号（带全局rob_id，单通道）
    val complete_o = Decoupled(new BallRsComplete)
  })

  // 简单的FIFO队列，只做缓冲
  val fifo = Module(new Queue(new Bundle {
    val cmd = new BallDecodeCmd
    val rob_id = UInt(log2Up(b.rob_entries).W)
  }, entries = 4))  // 小缓冲即可

// -----------------------------------------------------------------------------
// 入站 - FIFO入队
// -----------------------------------------------------------------------------
  fifo.io.enq <> io.ball_decode_cmd_i

// -----------------------------------------------------------------------------
// 出站 - 指令发射 (根据bid分发到对应Ball设备)
// -----------------------------------------------------------------------------
  val headEntry = fifo.io.deq.bits

  // 为每个Ball设备设置发射信号
  for (i <- 0 until numBalls) {
    val ballId = BallRsRegists(i).ballId.U
    io.issue_o.balls(i).valid := fifo.io.deq.valid && headEntry.cmd.bid === ballId
    io.issue_o.balls(i).bits.cmd := headEntry.cmd
    io.issue_o.balls(i).bits.rob_id := headEntry.rob_id
  }

  // FIFO deq.ready - 只有目标Ball设备ready时才能出队
  fifo.io.deq.ready := VecInit(
    BallRsRegists.zipWithIndex.map { case (info, idx) =>
      (headEntry.cmd.bid === info.ballId.U) && io.issue_o.balls(idx).ready
    }
  ).asUInt.orR

// -----------------------------------------------------------------------------
// 完成信号处理 - 直接转发给全局RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), numBalls))

  // 连接所有Ball设备的完成信号到仲裁器
  for (i <- 0 until numBalls) {
    completeArb.io.in(i).valid := io.commit_i.balls(i).valid
    completeArb.io.in(i).bits  := io.commit_i.balls(i).bits.rob_id
    io.commit_i.balls(i).ready := completeArb.io.in(i).ready
  }

  // 转发完成信号（带全局rob_id）
  io.complete_o.valid := completeArb.io.out.valid
  io.complete_o.bits.rob_id := completeArb.io.out.bits
  completeArb.io.out.ready := io.complete_o.ready
}
