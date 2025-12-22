package prototype.conv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status
import prototype.conv.configs.ConvConfig

/**
 * NVDLAConvBlackBox - BlackBox wrapper for NVDLA CONV module
 * Uses inline verilog to embed NVDLA CSC module
 */
class NVDLAConvBlackBox extends BlackBox with HasBlackBoxInline {

  val io = IO(new Bundle {
    val clock = Input(Clock())
    val reset = Input(Bool())

    // Simplified CONV interface
    val start = Input(Bool())
    val done  = Output(Bool())

    // Input feature map address
    val ifmap_addr  = Input(UInt(32.W))
    // Weight address
    val weight_addr = Input(UInt(32.W))
    // Output feature map address
    val ofmap_addr  = Input(UInt(32.W))

    // Convolution parameters
    val in_height    = Input(UInt(16.W))
    val in_width     = Input(UInt(16.W))
    val in_channels  = Input(UInt(16.W))
    val out_channels = Input(UInt(16.W))
    val kernel_h     = Input(UInt(8.W))
    val kernel_w     = Input(UInt(8.W))
    val stride_h     = Input(UInt(8.W))
    val stride_w     = Input(UInt(8.W))
    val pad_h        = Input(UInt(8.W))
    val pad_w        = Input(UInt(8.W))

    // Data width
    val data_width = Input(UInt(8.W))
  })

  setInline(
    "NVDLAConvBlackBox.v",
    s"""
       |module NVDLAConvBlackBox(
       |  input clock,
       |  input reset,
       |  input start,
       |  output reg done,
       |  input [31:0] ifmap_addr,
       |  input [31:0] weight_addr,
       |  input [31:0] ofmap_addr,
       |  input [15:0] in_height,
       |  input [15:0] in_width,
       |  input [15:0] in_channels,
       |  input [15:0] out_channels,
       |  input [7:0] kernel_h,
       |  input [7:0] kernel_w,
       |  input [7:0] stride_h,
       |  input [7:0] stride_w,
       |  input [7:0] pad_h,
       |  input [7:0] pad_w,
       |  input [7:0] data_width
       |);
       |
       |  reg [31:0] cycle_count;
       |  reg running;
       |
       |  always @(posedge clock) begin
       |    if (reset) begin
       |      done <= 1'b0;
       |      cycle_count <= 32'b0;
       |      running <= 1'b0;
       |    end else begin
       |      if (start && !running) begin
       |        running <= 1'b1;
       |        cycle_count <= 32'b0;
       |        done <= 1'b0;
       |      end else if (running) begin
       |        // Simplified: compute cycles based on convolution size
       |        // This is a placeholder - actual NVDLA CSC would be instantiated here
       |        if (cycle_count >= (in_height * in_width * kernel_h * kernel_w * in_channels * out_channels / 64)) begin
       |          done <= 1'b1;
       |          running <= 1'b0;
       |        end else begin
       |          cycle_count <= cycle_count + 1;
       |        end
       |      end
       |    end
       |  end
       |
       |endmodule
    """.stripMargin
  )
}

/**
 * Conv - Convolution computation unit
 * Simplified wrapper around NVDLA CONV module
 * Reads input feature map and weights from scratchpad, performs convolution, writes output
 */
@instantiable
class Conv(val parameter: ConvConfig)(implicit p: Parameters) extends Module with SerializableModule[ConvConfig] {
  // Get parameters from config
  val ballParam  = parameter.ballParam
  val InputNum   = parameter.InputNum
  val inputWidth = parameter.inputWidth
  val accWidth   = 32
  val bankWidth  = parameter.bankWidth

  @public
  val io = IO(new Bundle {
    // Command interface
    val cmdReq  = Flipped(Decoupled(new BallRsIssue(ballParam)))
    val cmdResp = Decoupled(new BallRsComplete(ballParam))

    // Unified bank read/write interface
    val bankRead  = Vec(ballParam.numBanks, Flipped(new SramReadIO(ballParam.bankEntries, bankWidth)))
    val bankWrite =
      Vec(ballParam.numBanks, Flipped(new SramWriteIO(ballParam.bankEntries, bankWidth, ballParam.bankMaskLen)))

    // Accumulator write interface (unified bank now)
    val bankWriteAcc =
      Vec(ballParam.numBanks, Flipped(new SramWriteIO(ballParam.bankEntries, accWidth, ballParam.bankMaskLen)))

    // Status output
    val status = new Status
  })

  // State machine
  val idle :: sLoadIfmap :: sLoadWeight :: sCompute :: sWrite :: complete :: Nil = Enum(6)
  val state                                                                      = RegInit(idle)

  // Instruction registers
  val robid_reg       = RegInit(0.U(10.W))
  val ifmap_addr_reg  = RegInit(0.U(10.W))
  val ifmap_bank_reg  = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val weight_addr_reg = RegInit(0.U(10.W))
  val weight_bank_reg = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val ofmap_addr_reg  = RegInit(0.U(10.W))
  val ofmap_bank_reg  = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val iter_reg        = RegInit(0.U(10.W))

  // Convolution parameters from special field (40 bits total)
  // special[15:0] = in_height (16 bits)
  // special[31:16] = in_width (16 bits)
  // special[39:32] = kernel_h (8 bits)
  // Note: kernel_w is encoded in lower 8 bits of kernel_h, or use a different encoding
  // For simplicity, we'll use kernel_h for both dimensions or extract from iter
  val in_height_reg = RegInit(0.U(16.W))
  val in_width_reg  = RegInit(0.U(16.W))
  val kernel_h_reg  = RegInit(0.U(8.W))
  val kernel_w_reg  = RegInit(0.U(8.W))

  // Counters
  val readCounter    = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val writeCounter   = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val computeCounter = RegInit(0.U(32.W))

  // NVDLA CONV BlackBox instance
  val nvdlaConv = Module(new NVDLAConvBlackBox)
  nvdlaConv.io.clock := clock
  nvdlaConv.io.reset := reset.asBool

  // Default SRAM assignments
  for (i <- 0 until ballParam.numBanks) {
    io.bankRead(i).req.valid        := false.B
    io.bankRead(i).req.bits.addr    := 0.U
    io.bankRead(i).req.bits.fromDMA := false.B
    io.bankRead(i).resp.ready       := false.B

    io.bankWrite(i).req.valid     := false.B
    io.bankWrite(i).req.bits.addr := 0.U
    io.bankWrite(i).req.bits.data := 0.U
    io.bankWrite(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(0.U(1.W)))
  }

  // Default accumulator assignments
  for (i <- 0 until ballParam.numBanks) {
    io.bankWriteAcc(i).req.valid     := false.B
    io.bankWriteAcc(i).req.bits.addr := 0.U
    io.bankWriteAcc(i).req.bits.data := 0.U
    io.bankWriteAcc(i).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(0.U(1.W)))
  }

  // Command interface defaults
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := robid_reg

  // NVDLA CONV interface defaults
  nvdlaConv.io.start        := false.B
  nvdlaConv.io.ifmap_addr   := ifmap_addr_reg
  nvdlaConv.io.weight_addr  := weight_addr_reg
  nvdlaConv.io.ofmap_addr   := ofmap_addr_reg
  nvdlaConv.io.in_height    := in_height_reg
  nvdlaConv.io.in_width     := in_width_reg
  nvdlaConv.io.in_channels  := 16.U // Default
  nvdlaConv.io.out_channels := 16.U // Default
  nvdlaConv.io.kernel_h     := kernel_h_reg
  nvdlaConv.io.kernel_w     := kernel_w_reg
  nvdlaConv.io.stride_h     := 1.U
  nvdlaConv.io.stride_w     := 1.U
  nvdlaConv.io.pad_h        := 0.U
  nvdlaConv.io.pad_w        := 0.U
  nvdlaConv.io.data_width   := inputWidth.U

  // Status output
  io.status.ready    := io.cmdReq.ready
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := (state === idle)
  io.status.init     := (state === sLoadIfmap) || (state === sLoadWeight)
  io.status.running  := (state === sCompute) || (state === sWrite)
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter     := computeCounter

  // State machine
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state          := sLoadIfmap
        readCounter    := 0.U
        writeCounter   := 0.U
        computeCounter := 0.U

        robid_reg       := io.cmdReq.bits.rob_id
        ifmap_addr_reg  := 0.U // New ISA: all operations start from row 0
        ifmap_bank_reg  := io.cmdReq.bits.cmd.op1_bank
        weight_addr_reg := 0.U // New ISA: all operations start from row 0
        weight_bank_reg := io.cmdReq.bits.cmd.op2_bank
        ofmap_addr_reg  := 0.U // New ISA: all operations start from row 0
        ofmap_bank_reg  := io.cmdReq.bits.cmd.wr_bank
        iter_reg        := io.cmdReq.bits.cmd.iter

        // Extract convolution parameters from special field (40 bits)
        in_height_reg := io.cmdReq.bits.cmd.special(15, 0)
        in_width_reg  := io.cmdReq.bits.cmd.special(31, 16)
        kernel_h_reg  := io.cmdReq.bits.cmd.special(39, 32)
        // kernel_w uses same value as kernel_h for simplicity, or could be encoded differently
        kernel_w_reg  := io.cmdReq.bits.cmd.special(39, 32)
      }
    }

    is(sLoadIfmap) {
      // Load input feature map (simplified: load one tile)
      when(readCounter < iter_reg) {
        io.bankRead(ifmap_bank_reg).req.valid        := true.B
        io.bankRead(ifmap_bank_reg).req.bits.addr    := ifmap_addr_reg + readCounter
        io.bankRead(ifmap_bank_reg).req.bits.fromDMA := false.B

        when(io.bankRead(ifmap_bank_reg).resp.valid) {
          io.bankRead(ifmap_bank_reg).resp.ready := true.B
          readCounter                            := readCounter + 1.U
        }
      }.otherwise {
        state       := sLoadWeight
        readCounter := 0.U
      }
    }

    is(sLoadWeight) {
      // Load weights (simplified: load one tile)
      when(readCounter < iter_reg) {
        io.bankRead(weight_bank_reg).req.valid        := true.B
        io.bankRead(weight_bank_reg).req.bits.addr    := weight_addr_reg + readCounter
        io.bankRead(weight_bank_reg).req.bits.fromDMA := false.B

        when(io.bankRead(weight_bank_reg).resp.valid) {
          io.bankRead(weight_bank_reg).resp.ready := true.B
          readCounter                             := readCounter + 1.U
        }
      }.otherwise {
        state              := sCompute
        readCounter        := 0.U
        nvdlaConv.io.start := true.B
      }
    }

    is(sCompute) {
      // Wait for NVDLA CONV to complete
      when(nvdlaConv.io.done) {
        state        := sWrite
        writeCounter := 0.U
      }.otherwise {
        computeCounter := computeCounter + 1.U
      }
    }

    is(sWrite) {
      // Write output feature map (simplified: write one tile)
      when(writeCounter < iter_reg) {
        io.bankWrite(ofmap_bank_reg).req.valid     := true.B
        io.bankWrite(ofmap_bank_reg).req.bits.addr := ofmap_addr_reg + writeCounter
        // Simplified: write zeros as placeholder (actual output would come from NVDLA CONV)
        io.bankWrite(ofmap_bank_reg).req.bits.data := 0.U
        io.bankWrite(ofmap_bank_reg).req.bits.mask := VecInit(Seq.fill(parameter.bankMaskLen)(1.U(1.W)))

        when(io.bankWrite(ofmap_bank_reg).req.ready) {
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
