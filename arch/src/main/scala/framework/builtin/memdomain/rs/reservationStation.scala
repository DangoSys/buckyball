package framework.builtin.memdomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain._
import framework.rocket.RoCCResponseBB


// Mem域的发射接口
class MemRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends RobEntry

// Mem域的完成接口
class MemRsComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

// Mem域发射接口组合 (Load + Store)
class MemIssueInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ld = Decoupled(new MemRsIssue)
  val st = Decoupled(new MemRsIssue)
}

// Mem域完成接口组合 (Load + Store)
class MemCommitInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ld = Flipped(Decoupled(new MemRsComplete))
  val st = Flipped(Decoupled(new MemRsComplete))
}

class MemReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // Mem Domain Decoder -> Rs
    val mem_decode_cmd_i = Flipped(new DecoupledIO(new MemDecodeCmd))
    val rs_rocc_o = new Bundle {      
      val resp  = new DecoupledIO(new RoCCResponseBB()(p))
      val busy  = Output(Bool())
    }
    // Rs -> MemLoader/MemStorer
    val issue_o     = new MemIssueInterface
    val commit_i    = new MemCommitInterface
  })

  val rob = Module(new ROB)

// -----------------------------------------------------------------------------
// 入站 - 指令分配
// -----------------------------------------------------------------------------
  rob.io.alloc <> io.mem_decode_cmd_i
  
// -----------------------------------------------------------------------------
// 出站 - 指令发射
// -----------------------------------------------------------------------------
  io.issue_o.ld.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_load
  io.issue_o.ld.bits  := Mux(io.issue_o.ld.valid, rob.io.issue.bits, DontCare)

  io.issue_o.st.valid := rob.io.issue.valid && rob.io.issue.bits.cmd.is_store
  io.issue_o.st.bits  := Mux(io.issue_o.st.valid, rob.io.issue.bits, DontCare) 
  
  rob.io.issue.ready  := (rob.io.issue.bits.cmd.is_load && io.issue_o.ld.ready) || 
                         (rob.io.issue.bits.cmd.is_store && io.issue_o.st.ready)
  
// -----------------------------------------------------------------------------
// 完成信号处理
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))
  completeArb.io.in(0).valid  := io.commit_i.ld.valid
  completeArb.io.in(0).bits   := Mux(io.commit_i.ld.valid, io.commit_i.ld.bits.rob_id, DontCare)
  completeArb.io.in(1).valid  := io.commit_i.st.valid  
  completeArb.io.in(1).bits   := Mux(io.commit_i.st.valid, io.commit_i.st.bits.rob_id, DontCare)
  
  rob.io.complete <> completeArb.io.out
  io.commit_i.ld.ready := completeArb.io.in(0).ready
  io.commit_i.st.ready := completeArb.io.in(1).ready
  
// -----------------------------------------------------------------------------
// 指令提交
// -----------------------------------------------------------------------------  
  // rob.io.commit.ready := true.B
  
// -----------------------------------------------------------------------------
// 响应生成
// -----------------------------------------------------------------------------
  io.rs_rocc_o.resp.valid := io.mem_decode_cmd_i.valid // 进来直接提交 
  io.rs_rocc_o.resp.bits  := DontCare
  io.rs_rocc_o.busy       := !rob.io.empty
}
