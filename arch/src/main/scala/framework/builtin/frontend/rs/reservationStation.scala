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

// Ball域的发射接口
class BallRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends RobEntry

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

class BallReservationStation(BallRsRegists: Seq[BallRsRegist])
  (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  
  val numBalls = BallRsRegists.length
  
  val io = IO(new Bundle {
    // Ball Domain Decoder -> Rs
    val ball_decode_cmd_i = Flipped(new DecoupledIO(new BallDecodeCmd))
    val rs_rocc_o = new Bundle {      
      val resp  = new DecoupledIO(new RoCCResponseBB()(p))
      val busy  = Output(Bool())
    }
    // Rs -> BallController
    val issue_o     = new BallIssueInterface(numBalls)
    val commit_i    = new BallCommitInterface(numBalls)
  })

  val rob = Module(new ROB)

// -----------------------------------------------------------------------------
// 入站 - 指令分配
// -----------------------------------------------------------------------------
  rob.io.alloc <> io.ball_decode_cmd_i
  
// -----------------------------------------------------------------------------
// 出站 - 指令发射 (根据指令类型分发到对应的Ball设备)
// -----------------------------------------------------------------------------
  // 创建ballId到索引的映射
  val ballIdToIndex = BallRsRegists.zipWithIndex.map { case (info, idx) => 
    info.ballId.U -> idx.U 
  }.toMap
  
  // 为每个Ball设备设置发射信号
  for (i <- 0 until numBalls) {
    val ballId = BallRsRegists(i).ballId.U
    io.issue_o.balls(i).valid := rob.io.issue.valid && rob.io.issue.bits.cmd.bid === ballId
    io.issue_o.balls(i).bits  := rob.io.issue.bits
  }

  // 设置ROB的ready信号 - 只有目标Ball设备ready时才能发射
  rob.io.issue.ready := VecInit(
    BallRsRegists.zipWithIndex.map { case (info, idx) =>
      (rob.io.issue.bits.cmd.bid === info.ballId.U) && io.issue_o.balls(idx).ready
    }
  ).asUInt.orR

  // -----------------------------------------------------------------------------
// 完成信号处理
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), numBalls))
  
  // 连接所有Ball设备的完成信号到仲裁器
  for (i <- 0 until numBalls) {
    completeArb.io.in(i).valid := io.commit_i.balls(i).valid
    completeArb.io.in(i).bits  := io.commit_i.balls(i).bits.rob_id
    io.commit_i.balls(i).ready := completeArb.io.in(i).ready
  }

  rob.io.complete <> completeArb.io.out
// -----------------------------------------------------------------------------
// 指令提交
// -----------------------------------------------------------------------------  
  // rob.io.commit.ready := true.B
  
// -----------------------------------------------------------------------------
// 响应生成
// -----------------------------------------------------------------------------
  io.rs_rocc_o.resp.valid := io.ball_decode_cmd_i.valid // 进来直接提交 
  io.rs_rocc_o.resp.bits  := DontCare
  io.rs_rocc_o.busy       := !rob.io.empty
}