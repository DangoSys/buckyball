package prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.stage._
import org.chipsalliance.cde.config.Parameters

import prototype.vector._
import framework.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.frontend.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.Status
import firrtl2.passes.CheckTypes.st


class Im2col(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    // cmd interface
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // Connect to Scratchpad SRAM read/write interface
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))

    // Status output
    val status = new Status
  })

  // State definitions
  val idle :: read :: read_and_convert :: complete :: Nil = Enum(4)
  // Current state register
  val state = RegInit(idle)
  // Conversion buffer
  val ConvertBuffer = RegInit(VecInit(Seq.fill(4)(VecInit(Seq.fill(b.veclane)(0.U(b.inputType.getWidth.W))))))
  // Row pointer marking top-left corner of convolution window
  val rowptr = RegInit(0.U(10.W))
  // Column pointer marking top-left corner of convolution window
  val colptr = RegInit(0.U(5.W))
  // Request counter in read state
  val reqcounter = RegInit(0.U(5.W))
  // Response counter in read state
  val respcounter = RegInit(0.U(5.W))
  // Store current instruction's RoB ID
  val robid_reg = RegInit(0.U(10.W))
  // Store kernel row count
  val krow_reg = RegInit(0.U(log2Up(b.veclane).W))
  // Store kernel column count
  val kcol_reg = RegInit(0.U(log2Up(b.veclane).W))
  // Store input matrix row count
  val inrow_reg = RegInit(0.U(10.W))
  // Store input matrix column count
  val incol_reg = RegInit(0.U((log2Up(b.veclane) + 1).W))
  // Store starting column number
  val startcol_reg = RegInit(0.U((log2Up(b.veclane) + 1).W))
  // Store starting row number
  val startrow_reg = RegInit(0.U(10.W))
  // Store write starting address
  val waddr_reg = RegInit(0.U(10.W))
  // Store write bank
  val wbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  // Store read starting address
  val raddr_reg = RegInit(0.U(10.W))
  // Store read bank
  val rbank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  // Batch iteration counter
  val iterCnt = RegInit(0.U(32.W))


  // SRAM default assignment
  for(i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid        := false.B
    io.sramRead(i).req.bits.addr    := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready       := (state === read) || (state === read_and_convert)
    io.sramWrite(i).req.valid       := false.B
    io.sramWrite(i).req.bits.addr   := 0.U
    io.sramWrite(i).req.bits.data   := 0.U
    io.sramWrite(i).req.bits.mask   := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
  }
  // cmd interface default assignment
  io.cmdReq.ready := true.B
  io.cmdResp.valid := false.B
  io.cmdResp.bits.rob_id := 0.U

  val rowcnt = rowptr - startrow_reg
  val colcnt = colptr - startcol_reg
  val rowmax = inrow_reg - krow_reg
  val colmax = incol_reg - kcol_reg

  switch(state) {
    // Idle state, waiting for instruction
    is(idle) {
      // Instruction arrives, initialize registers
      when(io.cmdReq.fire) {
        state      := read
        rowptr     := io.cmdReq.bits.cmd.special(37,28)
        colptr     := io.cmdReq.bits.cmd.special(27,23)
        reqcounter := 0.U
        respcounter:= 0.U
        // Kernel column count
        kcol_reg   := io.cmdReq.bits.cmd.special(3,0)
        // Kernel row count
        krow_reg   := io.cmdReq.bits.cmd.special(7,4)
        // Input matrix column count
        incol_reg  := io.cmdReq.bits.cmd.special(12,8)
        // Input matrix row count
        inrow_reg  := io.cmdReq.bits.cmd.special(22,13)
        // Starting column number
        startcol_reg := io.cmdReq.bits.cmd.special(27,23)
        // Starting row number
        startrow_reg := io.cmdReq.bits.cmd.special(37,28)
        robid_reg  := io.cmdReq.bits.rob_id
        waddr_reg  := io.cmdReq.bits.cmd.op2_bank_addr
        wbank_reg  := io.cmdReq.bits.cmd.op2_bank
        raddr_reg  := io.cmdReq.bits.cmd.op1_bank_addr
        rbank_reg  := io.cmdReq.bits.cmd.op1_bank
      }
    }
    // Read part of data, fill ConvertBuffer
    is(read) {
      // Send read request
      when(reqcounter < krow_reg) {
        reqcounter                           := reqcounter + 1.U
        io.sramRead(rbank_reg).req.valid     := true.B
        io.sramRead(rbank_reg).req.bits.addr := raddr_reg + reqcounter + startrow_reg
      }
      // Process read response and store in ConvertBuffer
      when(io.sramRead(rbank_reg).resp.fire) {
        ConvertBuffer(respcounter)           := io.sramRead(rbank_reg).resp.bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
        respcounter                          := respcounter + 1.U
      }
      // Determine whether to transition state
      state := Mux(respcounter === krow_reg, read_and_convert, read)

    }
    // Convert data and read remaining data, write back to spad
    is(read_and_convert) {
      // Move pointer
      when(colptr <= colmax && rowptr <= rowmax) {
        colptr := Mux(colptr === colmax, startcol_reg, colptr + 1.U)
        io.sramWrite(wbank_reg).req.valid     := true.B
        io.sramWrite(wbank_reg).req.bits.addr := waddr_reg + rowcnt * (colmax + 1.U - startcol_reg) + colcnt
        io.sramWrite(wbank_reg).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(~0.U(1.W)))
        io.sramWrite(wbank_reg).req.bits.data := {

          val window = Wire(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
          // Initialize all to 0 first
          for (i <- 0 until b.veclane) {
              window(i) := 0.U
          }

          // Fill window data
          for (i <- 0 until 4; j <- 0 until 4) {
            when(i.U < krow_reg && j.U < kcol_reg) {
              val bufferRow = (rowcnt + i.U) % krow_reg
              val bufferCol = (colptr + j.U) % incol_reg
              window((i.U * kcol_reg) + j.U) := ConvertBuffer(bufferRow)(bufferCol)
            }.otherwise {
              window((i.U * kcol_reg) + j.U) := 0.U
            }
          }

          // Rearrange data
          // For example, for klen_reg=3, combine (00)(01)(02)(10)(11)(12)(20)(21)(22)
          Cat((0 until b.veclane).map(i => window(i)).reverse)
        }
      }
      // Send read request early
      when(colptr === colmax - 1.U){
        io.sramRead(rbank_reg).req.valid     := true.B
        io.sramRead(rbank_reg).req.bits.addr := raddr_reg + krow_reg + rowptr
      }
      // Process read response and store in ConvertBuffer
      when(io.sramRead(rbank_reg).resp.fire){
        ConvertBuffer(rowcnt % krow_reg)     := io.sramRead(rbank_reg).resp.bits.data.asTypeOf(Vec(b.veclane, UInt(b.inputType.getWidth.W)))
        rowptr                               := rowptr + 1.U
      }
      // Determine whether to transition state
      state := Mux(rowptr === rowmax && colptr === colmax, complete, read_and_convert)
    }
    // Complete state, send completion signal
    is(complete) {
     io.cmdResp.valid       := true.B
     io.cmdResp.bits.rob_id := robid_reg
     state                  := idle
     when(io.cmdResp.fire) {
       iterCnt := iterCnt + 1.U
     }
    }
  }

  // Status signals
  io.status.ready := io.cmdReq.ready
  io.status.valid := io.cmdResp.valid
  io.status.idle := (state === idle)
  io.status.init := (state === read)
  io.status.running := (state === read_and_convert)
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter := iterCnt
}
