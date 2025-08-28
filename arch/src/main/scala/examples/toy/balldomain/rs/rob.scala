package examples.toy.balldomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.BallDecodeCmd

// ROB 条目数据结构 - 保留ROB ID支持乱序完成
class RobEntry(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd    = new BallDecodeCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

class ROB (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // 分配接口 
    val alloc = Flipped(new DecoupledIO(new BallDecodeCmd))
    
    // 发射接口 - 发射未完成的头部指令
    val issue = new DecoupledIO(new RobEntry)
    
    // 完成接口 - 报告指令完成
    val complete = Flipped(new DecoupledIO(UInt(log2Up(b.rob_entries).W)))
    
    // 提交接口 - 提交已完成的头部指令
    // val commit = new DecoupledIO(new RobEntry)
    
    // 状态信号
    val empty = Output(Bool())
    val full  = Output(Bool())
  })

  // 只使用 FIFO + 完成状态表, 只做入队出队，顺序执行顺序完成
  val robFifo = Module(new Queue(new RobEntry, b.rob_entries))
  val robIdCounter = RegInit(0.U(log2Up(b.rob_entries).W))
  val robTable = Reg(Vec(b.rob_entries, Bool()))
  
  // 初始化完成状态表
  for (i <- 0 until b.rob_entries) {
    when(reset.asBool) {
      robTable(i) := true.B
    }
  }

// -----------------------------------------------------------------------------
// 入站 - 指令分配
// -----------------------------------------------------------------------------
  robFifo.io.enq.valid       := io.alloc.valid
  robFifo.io.enq.bits.cmd    := io.alloc.bits
  robFifo.io.enq.bits.rob_id := robIdCounter
  
  io.alloc.ready := robFifo.io.enq.ready
  
  when(io.alloc.fire) {
    robIdCounter := robIdCounter + 1.U
    robTable(robIdCounter) := false.B
  }

// -----------------------------------------------------------------------------
// 完成信号处理 使用robTable跟踪
// -----------------------------------------------------------------------------
  io.complete.ready := true.B
  when(io.complete.fire) {
    robTable(io.complete.bits) := true.B
  }

// -----------------------------------------------------------------------------
// 出站 - 头部指令发射
// -----------------------------------------------------------------------------
  val headEntry     = robFifo.io.deq.bits
  val headCompleted = robTable(headEntry.rob_id)
  io.issue.valid   := robFifo.io.deq.valid && !headCompleted
  io.issue.bits    := headEntry

  robFifo.io.deq.ready := io.issue.ready && !headCompleted


// -----------------------------------------------------------------------------
// 状态信号
// -----------------------------------------------------------------------------
  val isEmpty = robTable.reduce(_ && _)
  val isFull = !robFifo.io.enq.ready
  
  io.empty := isEmpty
  io.full  := isFull
}