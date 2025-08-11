package buckyball.frontend.rs

import chisel3._
import chisel3.util._
import freechips.rocketchip.util.PlusArg
import org.chipsalliance.cde.config.Parameters
import buckyball.BBISA._
import buckyball.util.Util._
import buckyball.BuckyBallConfig
import buckyball.frontend.{PostDecodeCmd, BuckyBallRawCmd}
import framework.ballcore.ballcore.RoCCResponseBB



class ReservationStationIssue(cmd_t: BuckyBallCmd, id_width: Int) extends Bundle {
  val cmd = Output(cmd_t.cloneType)
  val rob_id = Output(UInt(id_width.W))
}

class ReservationStationComplete(id_width: Int) extends Bundle {
  val rob_id = UInt(id_width.W)
}

class RSISSInterface(cmd_t: BuckyBallCmd, id_width: Int) extends Bundle {
  val ld = Decoupled(new ReservationStationIssue(cmd_t, id_width))
  val st = Decoupled(new ReservationStationIssue(cmd_t, id_width))
  val ex = Decoupled(new ReservationStationIssue(cmd_t, id_width))
}

class RSCMTInterface(id_width: Int) extends Bundle {
  val ld = Flipped(Decoupled(new ReservationStationComplete(id_width)))
  val st = Flipped(Decoupled(new ReservationStationComplete(id_width)))
  val ex = Flipped(Decoupled(new ReservationStationComplete(id_width)))
}

class ReservationStation(implicit bbconfig: BuckyBallConfig, p: Parameters) extends Module {
  val queue_entries = 32
  val cmd_t = new BuckyBallCmd
  val rob_id_width = log2Up(bbconfig.rob_entries)

  val io = IO(new Bundle {
    // RAW CMD -> PostDecodeCmd -> BuckyBallCmd
    // ================================ 连接前端    
    val id_i = Flipped(Decoupled(new PostDecodeCmd))
    val rs_rocc_o = new Bundle {      
      val resp      = Decoupled(new RoCCResponseBB()(p))
      val busy      = Output(Bool())  // 是否有指令在ROB中等待提交
    }
    // ================================ 连接后端    
    val issue_o  = new RSISSInterface(cmd_t, rob_id_width)
    val commit_i = new RSCMTInterface(rob_id_width)
  })

  val ROB          = Module(new ReorderBuffer)
  val RobIdCounter = Module(new NextROBIdCounter)
  val ISSQueue     = Module(new IssueQueue(queue_entries))
  val CMTQueue     = Module(new CommitQueue(queue_entries))

  // id -> RobIdCounter -> ROB 先注册到ROB中
  RobIdCounter.io.post_decode_cmd_i <> io.id_i
  ROB.io.post_indexed_cmd_i <> RobIdCounter.io.post_index_cmd_o

  // ROB -> ISSQueue -> Controller
  ISSQueue.io.rob_issue_i <> ROB.io.issue_o
  io.issue_o <> ISSQueue.io.issue_o

  // controller -> CMTQueue -> ROB -> RoCC  
  CMTQueue.io.controller_commit_i <> io.commit_i
  ROB.io.commit_i <> CMTQueue.io.complete_o


  io.rs_rocc_o.resp <> ROB.io.rob_cmt_o.resp
  io.rs_rocc_o.busy := ROB.io.rob_cmt_o.busy

  // ROB -> RobIdCounter
  RobIdCounter.io.rob_cmt_i <> ROB.io.rob_robcnt_o
}
