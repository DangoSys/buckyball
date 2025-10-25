package framework.builtin.memdomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain._
import framework.rocket.RoCCResponseBB


// Mem域的发射接口 - 包含全局rob_id
class MemRsIssue(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd = new MemDecodeCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)  // 全局ROB ID
}

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

// 局部Mem保留站 - 简单的FIFO调度器
class MemReservationStation(implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // 解码后的指令输入（带全局rob_id）
    val mem_decode_cmd_i = Flipped(new DecoupledIO(new Bundle {
      val cmd = new MemDecodeCmd
      val rob_id = UInt(log2Up(b.rob_entries).W)  // 全局ROB ID
    }))

    // Rs -> MemLoader/MemStorer
    val issue_o     = new MemIssueInterface
    val commit_i    = new MemCommitInterface

    // 输出完成信号（带全局rob_id，单通道）
    val complete_o = Decoupled(new MemRsComplete)
  })

  // 简单的FIFO队列，只做缓冲
  val fifo = Module(new Queue(new Bundle {
    val cmd = new MemDecodeCmd
    val rob_id = UInt(log2Up(b.rob_entries).W)
  }, entries = 4))  // 小缓冲即可

// -----------------------------------------------------------------------------
// 入站 - FIFO入队
// -----------------------------------------------------------------------------
  fifo.io.enq <> io.mem_decode_cmd_i

// -----------------------------------------------------------------------------
// 出站 - 指令发射 (根据is_load/is_store分发)
// -----------------------------------------------------------------------------
  val headEntry = fifo.io.deq.bits

  // Load发射
  io.issue_o.ld.valid := fifo.io.deq.valid && headEntry.cmd.is_load
  io.issue_o.ld.bits.cmd := headEntry.cmd
  io.issue_o.ld.bits.rob_id := headEntry.rob_id

  // Store发射
  io.issue_o.st.valid := fifo.io.deq.valid && headEntry.cmd.is_store
  io.issue_o.st.bits.cmd := headEntry.cmd
  io.issue_o.st.bits.rob_id := headEntry.rob_id

  // FIFO deq.ready - 只有目标单元ready时才能出队
  fifo.io.deq.ready :=
    (headEntry.cmd.is_load  && io.issue_o.ld.ready) ||
    (headEntry.cmd.is_store && io.issue_o.st.ready)

// -----------------------------------------------------------------------------
// 完成信号处理 - 直接转发给全局RS
// -----------------------------------------------------------------------------
  val completeArb = Module(new Arbiter(UInt(log2Up(b.rob_entries).W), 2))

  completeArb.io.in(0).valid  := io.commit_i.ld.valid
  completeArb.io.in(0).bits   := io.commit_i.ld.bits.rob_id
  io.commit_i.ld.ready := completeArb.io.in(0).ready

  completeArb.io.in(1).valid  := io.commit_i.st.valid
  completeArb.io.in(1).bits   := io.commit_i.st.bits.rob_id
  io.commit_i.st.ready := completeArb.io.in(1).ready

  // 转发完成信号（带全局rob_id）
  io.complete_o.valid := completeArb.io.out.valid
  io.complete_o.bits.rob_id := completeArb.io.out.bits
  completeArb.io.out.ready := io.complete_o.ready
}
