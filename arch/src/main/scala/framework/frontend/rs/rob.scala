package framework.frontend.rs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyballConfigs.CustomBuckyballConfig
import examples.toy.balldomain.BallDecodeCmd

// ROB entry data structure - preserves ROB ID to support out-of-order completion
class RobEntry(implicit b: CustomBuckyballConfig, p: Parameters) extends Bundle {
  val cmd    = new BallDecodeCmd
  val rob_id = UInt(log2Up(b.rob_entries).W)
}

class ROB (implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val io = IO(new Bundle {
    // Allocation interface
    val alloc = Flipped(new DecoupledIO(new BallDecodeCmd))
    // Externally specified rob_id
    val alloc_rob_id = Input(UInt(log2Up(b.rob_entries).W))

    // Issue interface - issue uncompleted head instruction
    val issue = new DecoupledIO(new RobEntry)

    // Completion interface - report instruction completion
    val complete = Flipped(new DecoupledIO(UInt(log2Up(b.rob_entries).W)))

    // Commit interface - commit completed head instruction
    // val commit = new DecoupledIO(new RobEntry)

    // Status signals - exposed to reservation station for decision making
    val empty = Output(Bool())
    val full  = Output(Bool())
    // head pointer position
    val head_ptr = Output(UInt(log2Up(b.rob_entries).W))
    // Number of issued but uncompleted instructions
    val issued_count = Output(UInt(log2Up(b.rob_entries + 1).W))
    // Whether each entry is valid
    val entry_valid = Output(Vec(b.rob_entries, Bool()))
    // Whether each entry is complete
    val entry_complete = Output(Vec(b.rob_entries, Bool()))
  })

  // Circular ROB structure
  // Initialize to zero to avoid X states in FPGA
  val robEntries = RegInit(VecInit(Seq.fill(b.rob_entries)(0.U.asTypeOf(new RobEntry))))
  // Whether entry is valid
  val robValid   = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B)))
  // Whether entry is issued
  val robIssued  = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B)))
  // Whether entry is complete
  val robComplete = RegInit(VecInit(Seq.fill(b.rob_entries)(false.B)))

  // Circular queue pointers
  // Points to oldest uncommitted instruction
  val headPtr = RegInit(0.U(log2Up(b.rob_entries).W))
  // Points to next position to allocate
  val tailPtr = RegInit(0.U(log2Up(b.rob_entries).W))
  // ROB ID circular counter
  val robIdCounter = RegInit(0.U(log2Up(b.rob_entries).W))

  // Number of issued but uncompleted instructions (used to limit issue)
  val issuedCount = RegInit(0.U(log2Up(b.rob_entries + 1).W))
  // Maximum issue limit: half of ROB depth
  val maxIssueLimit = (b.rob_entries / 2).U

  // Queue status
  val isEmpty = headPtr === tailPtr && !robValid(headPtr)
  val isFull  = headPtr === tailPtr && robValid(headPtr)
  val count = Mux(isFull, b.rob_entries.U,
                  Mux(tailPtr >= headPtr, tailPtr - headPtr,
                      b.rob_entries.U + tailPtr - headPtr))

// -----------------------------------------------------------------------------
// Inbound - instruction allocation
// -----------------------------------------------------------------------------
  io.alloc.ready := !isFull

  when(io.alloc.fire) {
    robEntries(tailPtr).cmd    := io.alloc.bits
    robEntries(tailPtr).rob_id := robIdCounter
    robValid(tailPtr)   := true.B
    robIssued(tailPtr)  := false.B
    robComplete(tailPtr) := false.B

    // Update tail pointer and rob_id counter (circular)
    tailPtr := Mux(tailPtr === (b.rob_entries - 1).U, 0.U, tailPtr + 1.U)
    robIdCounter := Mux(robIdCounter === (b.rob_entries - 1).U, 0.U, robIdCounter + 1.U)
  }

// -----------------------------------------------------------------------------
// Completion signal processing
// -----------------------------------------------------------------------------
  io.complete.ready := true.B
  when(io.complete.fire) {
    val completeId = io.complete.bits
    robComplete(completeId) := true.B
    // When complete, decrement issued count
    when(robIssued(completeId)) {
      issuedCount := issuedCount - 1.U
    }
  }

// -----------------------------------------------------------------------------
// Outbound - issue instructions in order (starting from head)
// -----------------------------------------------------------------------------
  // Find first valid and unissued instruction starting from head
  val canIssue = Wire(Bool())
  val issuePtr = Wire(UInt(log2Up(b.rob_entries).W))

  // Default values
  canIssue := false.B
  issuePtr := headPtr

  // Scan from head to find first issuable instruction
  val scanValid = Wire(Vec(b.rob_entries, Bool()))
  for (i <- 0 until b.rob_entries) {
    val ptr = Mux(headPtr + i.U >= b.rob_entries.U,
                  headPtr + i.U - b.rob_entries.U,
                  headPtr + i.U)
    scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
  }

  // Find first issuable position
  val firstValid = PriorityEncoder(scanValid.asUInt)
  val hasValid = scanValid.asUInt.orR

  val actualIssuePtr = Mux(headPtr + firstValid >= b.rob_entries.U,
                           headPtr + firstValid - b.rob_entries.U,
                           headPtr + firstValid)

  // Can only issue if issue limit is not reached
  val canIssueMore = issuedCount < maxIssueLimit
  canIssue := hasValid && canIssueMore
  issuePtr := actualIssuePtr

  io.issue.valid := canIssue
  io.issue.bits  := robEntries(issuePtr)

  when(io.issue.fire) {
    robIssued(issuePtr) := true.B
    issuedCount := issuedCount + 1.U
  }

// -----------------------------------------------------------------------------
// Instruction commit - commit all completed instructions out-of-order
// -----------------------------------------------------------------------------
  // When head instruction completes, automatically commit and move head pointer
  // when(robValid(headPtr) && robComplete(headPtr)) {
  //   robValid(headPtr) := false.B
  //   robIssued(headPtr) := false.B
  //   robComplete(headPtr) := false.B
  //   headPtr := Mux(headPtr === (b.rob_entries - 1).U, 0.U, headPtr + 1.U)
  // } // Sequential commit version

  // Commit all completed instructions
  for (i <- 0 until b.rob_entries) {
    when(robValid(i.U) && robComplete(i.U)) {
      robValid(i.U) := false.B
      robIssued(i.U) := false.B
      robComplete(i.U) := false.B
    }
  }

  // Update head pointer: skip all completed (about to be cleared) positions
  // Find first "valid and incomplete" instruction position starting from head
  val nextHeadCandidates = Wire(Vec(b.rob_entries, Bool()))
  for (i <- 0 until b.rob_entries) {
    val ptr = Mux(headPtr + i.U >= b.rob_entries.U,
                  headPtr + i.U - b.rob_entries.U,
                  headPtr + i.U)
    // Entry is valid and incomplete (will not be committed)
    nextHeadCandidates(i) := robValid(ptr) && !robComplete(ptr)
  }

  val hasUncommitted = nextHeadCandidates.asUInt.orR
  val nextHeadOffset = PriorityEncoder(nextHeadCandidates.asUInt)
  val nextHeadPtr = Mux(headPtr + nextHeadOffset >= b.rob_entries.U,
                        headPtr + nextHeadOffset - b.rob_entries.U,
                        headPtr + nextHeadOffset)

  // Update head pointer:
  // - If there are uncompleted instructions, move head to first uncompleted position
  // - If there are no uncompleted instructions (all complete), move head to tail (ROB is empty)
  headPtr := Mux(hasUncommitted, nextHeadPtr, tailPtr)

// -----------------------------------------------------------------------------
// Status signals - exposed to reservation station
// -----------------------------------------------------------------------------
  io.empty := isEmpty
  io.full  := isFull
  io.head_ptr := headPtr
  io.issued_count := issuedCount
  io.entry_valid := robValid
  io.entry_complete := robComplete
}
