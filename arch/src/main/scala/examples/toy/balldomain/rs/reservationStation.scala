package examples.toy.balldomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain._
import framework.rocket.RoCCResponseBB


// Ball域的发射接口
class BallRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends RobEntry

// Ball域的完成接口
class BallRsComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// Ball域发射接口 (ball1: VecUnit, ball2: BBFP)
class BallIssueInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ball1 = Decoupled(new BallRsIssue)  // VecUnit
  val ball2 = Decoupled(new BallRsIssue)  // BBFP
}

// Ball域完成接口
class BallCommitInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ball1 = Flipped(Decoupled(new BallRsComplete))  // VecUnit
  val ball2 = Flipped(Decoupled(new BallRsComplete))  // BBFP
}

class BallReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // Ball Domain Decoder -> Rs
    val ball_decode_cmd_i = Flipped(new DecoupledIO(new BallDecodeCmd))
    val rs_rocc_o = new Bundle {      
      val resp  = new DecoupledIO(new RoCCResponseBB()(p))
      val busy  = Output(Bool())
    }
    // Rs -> BallController
    val issue_o     = new BallIssueInterface
    val commit_i    = new BallCommitInterface
  })

  val rob = Module(new ROB)

// -----------------------------------------------------------------------------
// 入站 - 指令分配
// -----------------------------------------------------------------------------
  rob.io.alloc <> io.ball_decode_cmd_i
  
// -----------------------------------------------------------------------------
// 出站 - 指令发射 (根据指令类型分发到ball1和ball2)
// -----------------------------------------------------------------------------
  // ball1 (VecUnit): is_vec指令
  io.issue_o.ball1.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_vec
  io.issue_o.ball1.bits  := rob.io.issue.bits
  
  // ball2 (BBFP): is_bbfp指令  
  io.issue_o.ball2.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_bbfp
  io.issue_o.ball2.bits  := rob.io.issue.bits
  
  rob.io.issue.ready := (rob.io.issue.bits.cmd.is_vec && io.issue_o.ball1.ready) || 
                        (rob.io.issue.bits.cmd.is_bbfp && io.issue_o.ball2.ready)
  
// -----------------------------------------------------------------------------
// 完成信号处理
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))
  completeArb.io.in(0).valid := io.commit_i.ball1.valid
  completeArb.io.in(0).bits  := io.commit_i.ball1.bits.rob_id
  completeArb.io.in(1).valid := io.commit_i.ball2.valid  
  completeArb.io.in(1).bits  := io.commit_i.ball2.bits.rob_id
  
  rob.io.complete <> completeArb.io.out
  io.commit_i.ball1.ready := completeArb.io.in(0).ready
  io.commit_i.ball2.ready := completeArb.io.in(1).ready
  
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