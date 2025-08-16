package examples.toy.balldomain

import chisel3._
import chisel3.util._
import freechips.rocketchip.util.PlusArg
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.util.Util._
import freechips.rocketchip.tile._

// EX域的发射接口
class ExReservationStationIssue(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id_width = log2Up(bbconfig.rob_entries)
  val cmd = Output(new ExBuckyBallCmd)
  val rob_id = Output(UInt(rob_id_width.W))
}

// EX域的完成接口
class ExReservationStationComplete(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id_width = log2Up(bbconfig.rob_entries)
  val rob_id = UInt(rob_id_width.W)
}

class ExReservationStation(implicit bbconfig: CustomBuckyBallConfig, p: Parameters) extends Module {
  val queue_entries = 32
  val rob_id_width = log2Up(bbconfig.rob_entries)

  val io = IO(new Bundle {
    // EX Domain Decoder -> ExReservationStation
    val ex_decode_cmd_i = Flipped(Decoupled(new ExDecodeCmd))
    val rs_rocc_o = new Bundle {      
      val resp = Decoupled(new RoCCResponse()(p))
      val busy = Output(Bool())
    }
    // ExReservationStation -> ExecuteController (使用ExBuckyBallCmd)
    val issue_o = Decoupled(new ExReservationStationIssue)
    val commit_i = Flipped(Decoupled(new ExReservationStationComplete))
  })

  // 简单的ROB和发射队列实现
  val cmdQueue = Module(new Queue(new ExDecodeCmd, queue_entries))
  val robIdCounter = RegInit(0.U(rob_id_width.W))
  
  // 指令入队
  cmdQueue.io.enq <> io.ex_decode_cmd_i
  
  // 指令发射
  io.issue_o.valid := cmdQueue.io.deq.valid
  cmdQueue.io.deq.ready := io.issue_o.ready
  io.issue_o.bits.cmd.ex_decode_cmd := cmdQueue.io.deq.bits
  io.issue_o.bits.cmd.rob_id := robIdCounter
  io.issue_o.bits.rob_id := robIdCounter
  
  when(io.issue_o.fire) {
    robIdCounter := robIdCounter + 1.U
  }
  
  // 简化的完成处理
  io.commit_i.ready := true.B
  io.rs_rocc_o.resp.valid := io.commit_i.valid
  io.rs_rocc_o.resp.bits := DontCare
  io.rs_rocc_o.busy := cmdQueue.io.count > 0.U
}