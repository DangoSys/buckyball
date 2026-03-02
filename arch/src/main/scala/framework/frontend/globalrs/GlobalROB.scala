package framework.frontend.globalrs

import chisel3._
import chisel3.util._
import chisel3.experimental._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.frontend.decoder.PostGDCmd
import framework.frontend.scoreboard.BankScoreboard

// DPI-C BlackBox for instruction trace
class ITraceDPI extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val is_issue  = Input(UInt(8.W))
    val rob_id    = Input(UInt(32.W))
    val domain_id = Input(UInt(32.W))
    val funct     = Input(UInt(32.W))
    val rs1       = Input(UInt(64.W))
    val rs2       = Input(UInt(64.W))
    val enable    = Input(Bool())
  })

  setInline(
    "ITraceDPI.v",
    """
      |import "DPI-C" function void dpi_itrace(
      |  input byte unsigned is_issue,
      |  input int unsigned rob_id,
      |  input int unsigned domain_id,
      |  input int unsigned funct,
      |  input longint unsigned rs1,
      |  input longint unsigned rs2
      |);
      |
      |module ITraceDPI(
      |  input [7:0] is_issue,
      |  input [31:0] rob_id,
      |  input [31:0] domain_id,
      |  input [31:0] funct,
      |  input [63:0] rs1,
      |  input [63:0] rs2,
      |  input enable
      |);
      |  always @(*) begin
      |    if (enable) begin
      |      dpi_itrace(is_issue, rob_id, domain_id, funct, rs1, rs2);
      |    end
      |  end
      |endmodule
    """.stripMargin
  )
}

@instantiable
class GlobalROB(val b: GlobalConfig) extends Module {

  val bankIdLen = b.frontend.bank_id_len

  @public
  val io = IO(new Bundle {
    // Allocation interface
    val alloc = Flipped(new DecoupledIO(new PostGDCmd(b)))

    // Issue interface - issue uncompleted head instruction
    val issue = new DecoupledIO(new GlobalRobEntry(b))

    // Completion interface - report instruction completion
    val complete = Flipped(new DecoupledIO(UInt(log2Up(b.frontend.rob_entries).W)))

    // Status signals - exposed to reservation station for decision making
    val empty          = Output(Bool())
    val full           = Output(Bool())
    // head pointer position
    val head_ptr       = Output(UInt(log2Up(b.frontend.rob_entries).W))
    // Number of issued but uncompleted instructions
    val issued_count   = Output(UInt(log2Up(b.frontend.rob_entries + 1).W))
    // Whether each entry is valid
    val entry_valid    = Output(Vec(b.frontend.rob_entries, Bool()))
    // Whether each entry is complete
    val entry_complete = Output(Vec(b.frontend.rob_entries, Bool()))
  })

  // Bank Scoreboard — instruction-agnostic hazard detection
  val scoreboard: Instance[BankScoreboard] = Instantiate(new BankScoreboard(b.memDomain.bankNum, b.frontend.rob_entries))

  // Instruction trace DPI-C module
  val itrace = Module(new ITraceDPI)
  itrace.io.is_issue  := 0.U
  itrace.io.rob_id    := 0.U
  itrace.io.domain_id := 0.U
  itrace.io.funct     := 0.U
  itrace.io.rs1       := 0.U
  itrace.io.rs2       := 0.U
  itrace.io.enable    := false.B

  // Circular ROB structure
  val robEntries    =
    RegInit(VecInit(Seq.fill(b.frontend.rob_entries)(0.U.asTypeOf(new GlobalRobEntry(b)))))
  // Whether entry is valid
  val robValid      = RegInit(VecInit(Seq.fill(b.frontend.rob_entries)(false.B)))
  // Whether entry is issued
  val robIssued     = RegInit(VecInit(Seq.fill(b.frontend.rob_entries)(false.B)))
  // Whether entry is complete
  val robComplete   = RegInit(VecInit(Seq.fill(b.frontend.rob_entries)(false.B)))
  // Points to oldest uncommitted instruction
  val headPtr       = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  // Points to next position to allocate
  val tailPtr       = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  // ROB ID circular counter
  val robIdCounter  = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  // Number of issued but uncompleted instructions (used to limit issue)
  val issuedCount   = RegInit(0.U(log2Up(b.frontend.rob_entries + 1).W))
  // Maximum issue limit: half of ROB depth
  val maxIssueLimit = (b.frontend.rob_entries / 2).U
  // Queue status
  val isEmpty       = headPtr === tailPtr && !robValid(headPtr)
  val isFull        = headPtr === tailPtr && robValid(headPtr)

  // Helper: circular pointer arithmetic
  def wrapPtr(v: UInt): UInt = Mux(v >= b.frontend.rob_entries.U, v - b.frontend.rob_entries.U, v)

// =============================================================================
// Inbound - instruction allocation
// Fence instructions are filtered out by GlobalReservationStation and never enter ROB.
// =============================================================================
  io.alloc.ready := !isFull

  when(io.alloc.fire) {
    robEntries(tailPtr).cmd    := io.alloc.bits
    robEntries(tailPtr).rob_id := robIdCounter
    robValid(tailPtr)          := true.B
    robIssued(tailPtr)         := false.B
    robComplete(tailPtr)       := false.B

    // Update tail pointer and rob_id counter (circular)
    tailPtr      := Mux(tailPtr === (b.frontend.rob_entries - 1).U, 0.U, tailPtr + 1.U)
    robIdCounter := Mux(robIdCounter === (b.frontend.rob_entries - 1).U, 0.U, robIdCounter + 1.U)
  }

// =============================================================================
// Completion signal processing
// =============================================================================
  io.complete.ready := true.B

  // Default: scoreboard complete not active
  scoreboard.complete.valid := false.B
  scoreboard.complete.bits  := 0.U.asTypeOf(scoreboard.complete.bits)

  when(io.complete.fire) {
    val completeId = io.complete.bits
    robComplete(completeId)   := true.B
    // When complete, decrement issued count
    when(robIssued(completeId)) {
      issuedCount := issuedCount - 1.U
    }
    // Update scoreboard: release bank resources
    scoreboard.complete.valid := true.B
    scoreboard.complete.bits  := robEntries(completeId).cmd.bankAccess

    // Instruction trace: complete
    itrace.io.is_issue  := 0.U
    itrace.io.rob_id    := completeId
    itrace.io.domain_id := robEntries(completeId).cmd.domain_id
    itrace.io.funct     := robEntries(completeId).cmd.cmd.funct
    itrace.io.rs1       := robEntries(completeId).cmd.cmd.rs1
    itrace.io.rs2       := robEntries(completeId).cmd.cmd.rs2
    itrace.io.enable    := true.B
  }

// =============================================================================
// Outbound - issue instructions with hazard detection
//
// Scan from head to find the first issuable instruction that:
//   1. is valid && !issued && !complete
//   2. has no bank hazard (checked via scoreboard)
// =============================================================================
  val canIssue = Wire(Bool())
  val issuePtr = Wire(UInt(log2Up(b.frontend.rob_entries).W))

  // Default values
  canIssue := false.B
  issuePtr := headPtr

  val scanValid = Wire(Vec(b.frontend.rob_entries, Bool()))
  for (i <- 0 until b.frontend.rob_entries) {
    val ptr = wrapPtr(headPtr + i.U)
    scanValid(i) := robValid(ptr) && !robIssued(ptr) && !robComplete(ptr)
  }

  // Find first candidate
  val firstValid = PriorityEncoder(scanValid.asUInt)
  val hasValid   = scanValid.asUInt.orR

  val actualIssuePtr = wrapPtr(headPtr + firstValid)

  // Scoreboard hazard query for the selected candidate
  scoreboard.query := robEntries(actualIssuePtr).cmd.bankAccess
  val noHazard = !scoreboard.hasHazard

  /* when(hasValid && scoreboard.hasHazard) { printf( "[ROB HAZARD] ptr=%d func7=%d rd0v=%d rd0=%d rd1v=%d rd1=%d wrv=%d
   * wr=%d\n", actualIssuePtr, robEntries(actualIssuePtr).cmd.cmd.funct,
   * robEntries(actualIssuePtr).cmd.bankAccess.rd_bank_0_valid, robEntries(actualIssuePtr).cmd.bankAccess.rd_bank_0_id,
   * robEntries(actualIssuePtr).cmd.bankAccess.rd_bank_1_valid, robEntries(actualIssuePtr).cmd.bankAccess.rd_bank_1_id,
   * robEntries(actualIssuePtr).cmd.bankAccess.wr_bank_valid, robEntries(actualIssuePtr).cmd.bankAccess.wr_bank_id ) } */

  // Can only issue if issue limit is not reached and no bank hazard
  val canIssueMore = issuedCount < maxIssueLimit
  canIssue := hasValid && canIssueMore && noHazard
  issuePtr := actualIssuePtr

  io.issue.valid := canIssue
  io.issue.bits  := robEntries(issuePtr)

  // Default: scoreboard issue not active
  scoreboard.issue.valid := false.B
  scoreboard.issue.bits  := 0.U.asTypeOf(scoreboard.issue.bits)

  when(io.issue.fire) {
    robIssued(issuePtr)    := true.B
    issuedCount            := issuedCount + 1.U
    // Update scoreboard: claim bank resources
    scoreboard.issue.valid := true.B
    scoreboard.issue.bits  := robEntries(issuePtr).cmd.bankAccess

    // Instruction trace: issue
    itrace.io.is_issue  := 1.U
    itrace.io.rob_id    := robEntries(issuePtr).rob_id
    itrace.io.domain_id := robEntries(issuePtr).cmd.domain_id
    itrace.io.funct     := robEntries(issuePtr).cmd.cmd.funct
    itrace.io.rs1       := robEntries(issuePtr).cmd.cmd.rs1
    itrace.io.rs2       := robEntries(issuePtr).cmd.cmd.rs2
    itrace.io.enable    := true.B
  }

// =============================================================================
// Instruction commit - commit all completed instructions out-of-order
// =============================================================================
  for (i <- 0 until b.frontend.rob_entries) {
    when(robValid(i.U) && robComplete(i.U)) {
      robValid(i.U)    := false.B
      robIssued(i.U)   := false.B
      robComplete(i.U) := false.B
    }
  }

  // Update head pointer: skip all completed (about to be cleared) positions
  val nextHeadCandidates = Wire(Vec(b.frontend.rob_entries, Bool()))
  for (i <- 0 until b.frontend.rob_entries) {
    val ptr = wrapPtr(headPtr + i.U)
    nextHeadCandidates(i) := robValid(ptr) && !robComplete(ptr)
  }

  val hasUncommitted = nextHeadCandidates.asUInt.orR
  val nextHeadOffset = PriorityEncoder(nextHeadCandidates.asUInt)
  val nextHeadPtr    = wrapPtr(headPtr + nextHeadOffset)

  headPtr := Mux(hasUncommitted, nextHeadPtr, tailPtr)

// =============================================================================
// Status signals
// =============================================================================
  io.empty          := isEmpty
  io.full           := isFull
  io.head_ptr       := headPtr
  io.issued_count   := issuedCount
  io.entry_valid    := robValid
  io.entry_complete := robComplete
}
