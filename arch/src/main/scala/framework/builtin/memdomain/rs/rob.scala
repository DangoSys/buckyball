package framework.builtin.memdomain.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import framework.builtin.memdomain.MemDecodeCmd

// ROB entry data structure - preserves ROB ID to support out-of-order completion
class RobEntry(implicit b: CustomBuckyBallConfig, p: Parameters) extends Bundle {
  val cmd    = new MemDecodeCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

class ROB (implicit b: CustomBuckyBallConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // Allocation interface
    val alloc = Flipped(new DecoupledIO(new MemDecodeCmd))

    // Issue interface - issue uncompleted head instruction
    val issue = new DecoupledIO(new RobEntry)

    // Completion interface - report instruction completion
    val complete = Flipped(new DecoupledIO(UInt(log2Up(b.rob_entries).W)))

    // Commit interface - commit completed head instruction
    // val commit = new DecoupledIO(new RobEntry)

    // Status signals
    val empty = Output(Bool())
    val full  = Output(Bool())
  })

  // Only use FIFO + completion status table, only enqueue/dequeue, sequential execution and sequential completion
  val robFifo = Module(new Queue(new RobEntry, b.rob_entries))
  val robIdCounter = RegInit(0.U(log2Up(b.rob_entries).W))
  // Initialize to false to avoid X states in FPGA
  val robTable = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B)))

  // Initialize completion status table
  for (i <- 0 until b.rob_entries) {
    when(reset.asBool) {
      robTable(i) := true.B
    }
  }

// -----------------------------------------------------------------------------
// Inbound - instruction allocation
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
// Completion signal processing using robTable tracking
// -----------------------------------------------------------------------------
  io.complete.ready := true.B
  when(io.complete.fire) {
    robTable(io.complete.bits) := true.B
  }

// -----------------------------------------------------------------------------
// Outbound - head instruction issue
// -----------------------------------------------------------------------------
  val headEntry     = robFifo.io.deq.bits
  val headCompleted = robTable(headEntry.rob_id)
  io.issue.valid   := robFifo.io.deq.valid && !headCompleted
  io.issue.bits    := headEntry

  robFifo.io.deq.ready := io.issue.ready && !headCompleted


// -----------------------------------------------------------------------------
// Status signals
// -----------------------------------------------------------------------------
  val isEmpty = robTable.reduce(_ && _)
  val isFull = !robFifo.io.enq.ready

  io.empty := isEmpty
  io.full  := isFull
}
