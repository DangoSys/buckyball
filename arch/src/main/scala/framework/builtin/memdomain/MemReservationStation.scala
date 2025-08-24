package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import freechips.rocketchip.util.PlusArg
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.util.Util._
import freechips.rocketchip.tile._
import framework.rocket.RoCCResponseBB

// Mem域的发射接口
class MemReservationStationIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id_width = log2Up(b.rob_entries)
  val cmd = Output(new MemBuckyBallCmd)
  val rob_id = Output(UInt(rob_id_width.W))
}

// Mem域的完成接口
class MemReservationStationComplete(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val rob_id_width = log2Up(b.rob_entries)
  val rob_id = UInt(rob_id_width.W)
}

// Mem域发射接口组合 (Load + Store)
class MemIssueInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ld = Decoupled(new MemReservationStationIssue)
  val st = Decoupled(new MemReservationStationIssue)
}

// Mem域完成接口组合 (Load + Store)
class MemCommitInterface(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val ld = Flipped(Decoupled(new MemReservationStationComplete))
  val st = Flipped(Decoupled(new MemReservationStationComplete))
}

class MemReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val queue_entries = 32
  val rob_id_width = log2Up(b.rob_entries)

  val io = IO(new Bundle {
    // Mem Domain Decoder -> MemReservationStation
    val mem_decode_cmd_i = Flipped(Decoupled(new MemDecodeCmd))
    val rs_rocc_o = new Bundle {      
      val resp = Decoupled(new RoCCResponseBB()(p))
      val busy = Output(Bool())
    }
    // MemReservationStation -> MemLoader/MemStorer
    val issue_o = new MemIssueInterface
    val commit_i = new MemCommitInterface
  })

  // 分离的队列用于Load和Store
  val loadQueue = Module(new Queue(new MemDecodeCmd, queue_entries))
  val storeQueue = Module(new Queue(new MemDecodeCmd, queue_entries))
  val robIdCounter = RegInit(0.U(rob_id_width.W))
  
  // 根据指令类型分发到不同队列
  loadQueue.io.enq.valid := io.mem_decode_cmd_i.valid && io.mem_decode_cmd_i.bits.is_load
  storeQueue.io.enq.valid := io.mem_decode_cmd_i.valid && io.mem_decode_cmd_i.bits.is_store
  loadQueue.io.enq.bits := io.mem_decode_cmd_i.bits
  storeQueue.io.enq.bits := io.mem_decode_cmd_i.bits
  
  io.mem_decode_cmd_i.ready := Mux(io.mem_decode_cmd_i.bits.is_load, 
    loadQueue.io.enq.ready, storeQueue.io.enq.ready)
  
  // Load发射
  io.issue_o.ld.valid := loadQueue.io.deq.valid
  loadQueue.io.deq.ready := io.issue_o.ld.ready
  io.issue_o.ld.bits.cmd.mem_decode_cmd := loadQueue.io.deq.bits
  io.issue_o.ld.bits.cmd.rob_id := robIdCounter
  io.issue_o.ld.bits.rob_id := robIdCounter
  
  // Store发射
  io.issue_o.st.valid := storeQueue.io.deq.valid
  storeQueue.io.deq.ready := io.issue_o.st.ready
  io.issue_o.st.bits.cmd.mem_decode_cmd := storeQueue.io.deq.bits
  io.issue_o.st.bits.cmd.rob_id := robIdCounter
  io.issue_o.st.bits.rob_id := robIdCounter
  
  when(io.issue_o.ld.fire || io.issue_o.st.fire) {
    robIdCounter := robIdCounter + 1.U
  }
  
  // 简化的响应处理
  io.commit_i.ld.ready := true.B
  io.commit_i.st.ready := true.B
  
  val respArb = Module(new Arbiter(new RoCCResponse()(p), 2))
  respArb.io.in(0).valid := io.commit_i.ld.valid
  respArb.io.in(1).valid := io.commit_i.st.valid
  respArb.io.in(0).bits := DontCare
  respArb.io.in(1).bits := DontCare
  io.rs_rocc_o.resp <> respArb.io.out
  
  io.rs_rocc_o.busy := loadQueue.io.count > 0.U || storeQueue.io.count > 0.U
}