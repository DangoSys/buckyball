package prototype.nagisa.matmul

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status

class PiDRAMmarcoBlackBox extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())

    // Control interface
    val start = Input(Bool())
    val done  = Output(Bool())

    // Data addresses
    val op1_addr    = Input(UInt(32.W))
    val op2_addr    = Input(UInt(32.W))
    val result_addr = Input(UInt(32.W))

    // Computation parameters
    val rows    = Input(UInt(16.W))
    val cols    = Input(UInt(16.W))
    val op_type = Input(UInt(4.W)) // 0: matmul, 1: add, 2: mul, etc.

    // Data width
    val data_width = Input(UInt(8.W))
  })

  setInline(
    "PiDRAMmarcoBlackBox.v",
    s"""
       |module PiDRAMmarcoBlackBox(
       |  input clock,
       |  input reset,
       |  input start,
       |  output reg done,
       |  input [31:0] op1_addr,
       |  input [31:0] op2_addr,
       |  input [31:0] result_addr,
       |  input [15:0] rows,
       |  input [15:0] cols,
       |  input [3:0] op_type,
       |  input [7:0] data_width
       |);
       |
       |  // State machine for marco computation
       |  reg [2:0] state;
       |  localparam IDLE = 3'b000;
       |  localparam LOAD = 3'b001;
       |  localparam COMPUTE = 3'b010;
       |  localparam STORE = 3'b011;
       |  localparam DONE = 3'b100;
       |
       |  // Cycle counter for computation
       |  reg [31:0] cycle_count;
       |  reg [31:0] total_cycles;
       |  reg running;
       |
       |  // Compute total cycles based on operation type and size
       |  always @(*) begin
       |    case(op_type)
       |      4'b0000: // Matrix multiplication: rows * cols * cols
       |        total_cycles = rows * cols * cols;
       |      4'b0001: // Element-wise addition: rows * cols
       |        total_cycles = rows * cols;
       |      4'b0010: // Element-wise multiplication: rows * cols
       |        total_cycles = rows * cols;
       |      default:
       |        total_cycles = rows * cols;
       |    endcase
       |  end
       |
       |  always @(posedge clock) begin
       |    if (reset) begin
       |      done <= 1'b0;
       |      state <= IDLE;
       |      cycle_count <= 32'b0;
       |      running <= 1'b0;
       |    end else begin
       |      case(state)
       |        IDLE: begin
       |          done <= 1'b0;
       |          cycle_count <= 32'b0;
       |          running <= 1'b0;
       |          if (start) begin
       |            state <= LOAD;
       |            running <= 1'b1;
       |          end
       |        end
       |        LOAD: begin
       |          // Load phase: simulate data loading from memory
       |          // In real marco, this would be handled by the memory controller
       |          state <= COMPUTE;
       |        end
       |        COMPUTE: begin
       |          // Compute phase: perform marco computation
       |          // This simulates the computation cycles in marco
       |          if (cycle_count >= total_cycles) begin
       |            state <= STORE;
       |            cycle_count <= 32'b0;
       |          end else begin
       |            cycle_count <= cycle_count + 1;
       |          end
       |        end
       |        STORE: begin
       |          // Store phase: write results back
       |          state <= DONE;
       |        end
       |        DONE: begin
       |          done <= 1'b1;
       |          running <= 1'b0;
       |          if (!start) begin
       |            state <= IDLE;
       |          end
       |        end
       |        default: begin
       |          state <= IDLE;
       |        end
       |      endcase
       |    end
       |  end
       |
       |endmodule
    """.stripMargin
  )
}

@instantiable
class marco(val parameter: BallDomainParam)(implicit p: Parameters)
    extends Module
    with SerializableModule[BallDomainParam] {
  // Derived parameters
  val InputNum   = 16
  val inputWidth = 8
  val accWidth   = 32
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    // Command interface
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(parameter)))
    val cmdResp = Decoupled(new BallRsComplete(parameter))

    // Scratchpad SRAM read/write interface
    val sramRead  = Vec(parameter.numBanks, Flipped(new SramReadIO(parameter.bankEntries, bankWidth)))
    val sramWrite =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, bankWidth, parameter.bankMaskLen)))

    // Accumulator write interface (unified bank now)
    val accWrite =
      Vec(parameter.numBanks, Flipped(new SramWriteIO(parameter.bankEntries, accWidth, parameter.bankMaskLen)))

    // Status output
    val status = new Status
  })

  // State machine
  val idle :: sLoadOp1 :: sLoadOp2 :: sCompute :: sWrite :: complete :: Nil = Enum(6)
  val state                                                                 = RegInit(idle)

  // Instruction registers
  val robid_reg       = RegInit(0.U(10.W))
  val op1_addr_reg    = RegInit(0.U(10.W))
  val op1_bank_reg    = RegInit(0.U(log2Up(parameter.numBanks).W))
  val op2_addr_reg    = RegInit(0.U(10.W))
  val op2_bank_reg    = RegInit(0.U(log2Up(parameter.numBanks).W))
  val result_addr_reg = RegInit(0.U(10.W))
  val result_bank_reg = RegInit(0.U(log2Up(parameter.numBanks).W))
  val iter_reg        = RegInit(0.U(10.W))

  // marco parameters from special field (40 bits total)
  // special[15:0] = rows (16 bits)
  // special[31:16] = cols (16 bits)
  // special[35:32] = op_type (4 bits): 0=matmul, 1=add, 2=mul
  val rows_reg    = RegInit(0.U(16.W))
  val cols_reg    = RegInit(0.U(16.W))
  val op_type_reg = RegInit(0.U(4.W))

  // Counters
  val readCounter    = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val writeCounter   = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val computeCounter = RegInit(0.U(32.W))

  // PiDRAM marco BlackBox instance
  val pidrammarco = Module(new PiDRAMmarcoBlackBox)
  pidrammarco.io.clock := clock
  pidrammarco.io.reset := reset.asBool

  // Default SRAM assignments
  for (i <- 0 until parameter.numBanks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := false.B

    io.sramWrite(i).req.valid     := false.B
    io.sramWrite(i).req.bits.addr := 0.U
    io.sramWrite(i).req.bits.data := 0.U
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(0.U(1.W)))
  }

  // Default accumulator assignments
  for (i <- 0 until parameter.numBanks) {
    io.accWrite(i).req.valid     := false.B
    io.accWrite(i).req.bits.addr := 0.U
    io.accWrite(i).req.bits.data := 0.U
    io.accWrite(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(0.U(1.W)))
  }

  // Command interface defaults
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := robid_reg

  // PiDRAM marco interface defaults
  pidrammarco.io.start       := false.B
  pidrammarco.io.op1_addr    := op1_addr_reg
  pidrammarco.io.op2_addr    := op2_addr_reg
  pidrammarco.io.result_addr := result_addr_reg
  pidrammarco.io.rows        := rows_reg
  pidrammarco.io.cols        := cols_reg
  pidrammarco.io.op_type     := op_type_reg
  pidrammarco.io.data_width  := inputWidth.U

  // Status output
  io.status.ready    := io.cmdReq.ready
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := (state === idle)
  io.status.init     := (state === sLoadOp1) || (state === sLoadOp2)
  io.status.running  := (state === sCompute) || (state === sWrite)
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter     := computeCounter

  // State machine
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state          := sLoadOp1
        readCounter    := 0.U
        writeCounter   := 0.U
        computeCounter := 0.U

        robid_reg       := io.cmdReq.bits.rob_id
        op1_addr_reg    := 0.U // New ISA: all operations start from row 0
        op1_bank_reg    := io.cmdReq.bits.cmd.op1_bank
        op2_addr_reg    := 0.U // New ISA: all operations start from row 0
        op2_bank_reg    := io.cmdReq.bits.cmd.op2_bank
        result_addr_reg := 0.U // New ISA: all operations start from row 0
        result_bank_reg := io.cmdReq.bits.cmd.wr_bank
        iter_reg        := io.cmdReq.bits.cmd.iter

        // Extract marco parameters from special field (40 bits)
        // special[15:0] = rows, special[31:16] = cols, special[35:32] = op_type
        rows_reg    := io.cmdReq.bits.cmd.special(15, 0)
        cols_reg    := io.cmdReq.bits.cmd.special(31, 16)
        op_type_reg := io.cmdReq.bits.cmd.special(35, 32)
      }
    }

    is(sLoadOp1) {
      // Load operand 1 (simplified: load one tile)
      when(readCounter < iter_reg) {
        io.sramRead(op1_bank_reg).req.valid        := true.B
        io.sramRead(op1_bank_reg).req.bits.addr    := op1_addr_reg + readCounter
        io.sramRead(op1_bank_reg).req.bits.fromDMA := false.B

        when(io.sramRead(op1_bank_reg).resp.valid) {
          io.sramRead(op1_bank_reg).resp.ready := true.B
          readCounter                          := readCounter + 1.U
        }
      }.otherwise {
        state       := sLoadOp2
        readCounter := 0.U
      }
    }

    is(sLoadOp2) {
      // Load operand 2 (simplified: load one tile)
      when(readCounter < iter_reg) {
        io.sramRead(op2_bank_reg).req.valid        := true.B
        io.sramRead(op2_bank_reg).req.bits.addr    := op2_addr_reg + readCounter
        io.sramRead(op2_bank_reg).req.bits.fromDMA := false.B

        when(io.sramRead(op2_bank_reg).resp.valid) {
          io.sramRead(op2_bank_reg).resp.ready := true.B
          readCounter                          := readCounter + 1.U
        }
      }.otherwise {
        state                := sCompute
        readCounter          := 0.U
        pidrammarco.io.start := true.B
      }
    }

    is(sCompute) {
      // Wait for PiDRAM marco to complete
      when(pidrammarco.io.done) {
        state        := sWrite
        writeCounter := 0.U
      }.otherwise {
        computeCounter := computeCounter + 1.U
      }
    }

    is(sWrite) {
      // Write result (simplified: write one tile)
      when(writeCounter < iter_reg) {
        io.sramWrite(result_bank_reg).req.valid     := true.B
        io.sramWrite(result_bank_reg).req.bits.addr := result_addr_reg + writeCounter
        // Simplified: write zeros as placeholder (actual output would come from PiDRAM marco)
        io.sramWrite(result_bank_reg).req.bits.data := 0.U
        io.sramWrite(result_bank_reg).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(1.U(1.W)))

        when(io.sramWrite(result_bank_reg).req.ready) {
          writeCounter := writeCounter + 1.U
        }
      }.otherwise {
        state := complete
      }
    }

    is(complete) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.ready) {
        state := idle
      }
    }
  }
}
