package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.prototype.vector._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.im2col.configs.Im2colBallParam

@instantiable
class Im2col(val b: GlobalConfig) extends Module {
  val ballConfig = Im2colBallParam()
  val InputNum   = ballConfig.InputNum
  val inputWidth = ballConfig.inputWidth
  val bankWidth  = b.memDomain.bankWidth

  // Get bandwidth from config
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  // State definitions
  val idle :: read :: read_and_convert :: complete :: Nil = Enum(4)
  // Current state register
  val state                                               = RegInit(idle)
  // Conversion buffer
  val ConvertBuffer                                       = RegInit(VecInit(Seq.fill(4)(VecInit(Seq.fill(InputNum)(0.U(inputWidth.W))))))
  // Row pointer marking top-left corner of convolution window
  val rowptr                                              = RegInit(0.U(10.W))
  // Column pointer marking top-left corner of convolution window
  val colptr                                              = RegInit(0.U(5.W))
  // Request counter in read state
  val reqcounter                                          = RegInit(0.U(5.W))
  // Response counter in read state
  val respcounter                                         = RegInit(0.U(5.W))
  // Store current instruction's RoB ID
  val rob_id_reg                                          = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  // Store kernel row count
  val krow_reg                                            = RegInit(0.U(log2Up(InputNum).W))
  // Store kernel column count
  val kcol_reg                                            = RegInit(0.U(log2Up(InputNum).W))
  // Store input matrix row count
  val inrow_reg                                           = RegInit(0.U(10.W))
  // Store input matrix column count
  val incol_reg                                           = RegInit(0.U((log2Up(InputNum) + 1).W))
  // Store starting column number
  val startcol_reg                                        = RegInit(0.U((log2Up(InputNum) + 1).W))
  // Store starting row number
  val startrow_reg                                        = RegInit(0.U(10.W))
  // Store write starting address
  val waddr_reg                                           = RegInit(0.U(10.W))
  // Store write bank
  val wbank_reg                                           = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  // Store read starting address
  val raddr_reg                                           = RegInit(0.U(10.W))
  // Store read bank
  val rbank_reg                                           = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  // Batch iteration counter
  val iterCnt                                             = RegInit(0.U(32.W))

  // SRAM default assignment
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := (state === read) || (state === read_and_convert)
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).rob_id           := rob_id_reg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).acc_group_id     := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := true.B
    io.bankWrite(i).bank_id           := wbank_reg
    io.bankWrite(i).rob_id            := rob_id_reg
    io.bankWrite(i).ball_id           := 0.U
    io.bankWrite(i).acc_group_id      := 0.U
  }
  // cmd interface default assignment
  io.cmdReq.ready := true.B
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  val rowcnt = rowptr - startrow_reg
  val colcnt = colptr - startcol_reg
  val rowmax = inrow_reg - krow_reg
  val colmax = incol_reg - kcol_reg

  switch(state) {
    // Idle state, waiting for instruction
    is(idle) {
      // Instruction arrives, initialize registers
      when(io.cmdReq.fire) {
        state        := read
        rowptr       := io.cmdReq.bits.cmd.special(37, 28)
        colptr       := io.cmdReq.bits.cmd.special(27, 23)
        reqcounter   := 0.U
        respcounter  := 0.U
        // Kernel column count
        kcol_reg     := io.cmdReq.bits.cmd.special(3, 0)
        // Kernel row count
        krow_reg     := io.cmdReq.bits.cmd.special(7, 4)
        // Input matrix column count
        incol_reg    := io.cmdReq.bits.cmd.special(12, 8)
        // Input matrix row count
        inrow_reg    := io.cmdReq.bits.cmd.special(22, 13)
        // Starting column number
        startcol_reg := io.cmdReq.bits.cmd.special(27, 23)
        // Starting row number
        startrow_reg := io.cmdReq.bits.cmd.special(37, 28)
        rob_id_reg   := io.cmdReq.bits.rob_id
        waddr_reg    := 0.U
        wbank_reg    := io.cmdReq.bits.cmd.op2_bank
        raddr_reg    := 0.U
        rbank_reg    := io.cmdReq.bits.cmd.op1_bank
      }
    }
    // Read part of data, fill ConvertBuffer
    is(read) {
      // Send read request
      when(reqcounter < krow_reg) {
        reqcounter                              := reqcounter + 1.U
        io.bankRead(rbank_reg).io.req.valid     := true.B
        io.bankRead(rbank_reg).io.req.bits.addr := raddr_reg + reqcounter + startrow_reg
      }
      // Process read response and store in ConvertBuffer
      when(io.bankRead(rbank_reg).io.resp.fire) {
        ConvertBuffer(respcounter) := io.bankRead(rbank_reg).io.resp.bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
        respcounter                := respcounter + 1.U
      }
      // Determine whether to transition state
      state := Mux(respcounter === krow_reg, read_and_convert, read)

    }
    // Convert data and read remaining data, write back to spad
    is(read_and_convert) {
      // Move pointer
      when(colptr <= colmax && rowptr <= rowmax) {
        colptr                                   := Mux(colptr === colmax, startcol_reg, colptr + 1.U)
        io.bankWrite(wbank_reg).io.req.valid     := true.B
        io.bankWrite(wbank_reg).io.req.bits.addr := waddr_reg + rowcnt * (colmax + 1.U - startcol_reg) + colcnt
        io.bankWrite(wbank_reg).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(~0.U(1.W)))
        io.bankWrite(wbank_reg).io.req.bits.data := {

          val window = Wire(Vec(InputNum, UInt(inputWidth.W)))
          // Initialize all to 0 first
          for (i <- 0 until InputNum) {
            window(i) := 0.U
          }

          // Fill window data
          for {
            i <- 0 until 4
            j <- 0 until 4
          } {
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
          Cat((0 until InputNum).map(i => window(i)).reverse)
        }
      }
      // Send read request early
      when(colptr === colmax - 1.U) {
        io.bankRead(rbank_reg).io.req.valid     := true.B
        io.bankRead(rbank_reg).io.req.bits.addr := raddr_reg + krow_reg + rowptr
      }
      // Process read response and store in ConvertBuffer
      when(io.bankRead(rbank_reg).io.resp.fire) {
        ConvertBuffer(rowcnt % krow_reg) := io.bankRead(rbank_reg).io.resp.bits.data.asTypeOf(Vec(
          InputNum,
          UInt(inputWidth.W)
        ))
        rowptr := rowptr + 1.U
      }
      // Determine whether to transition state
      state := Mux(rowptr === rowmax && colptr === colmax, complete, read_and_convert)
    }
    // Complete state, send completion signal
    is(complete) {
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
      state                  := idle
      when(io.cmdResp.fire) {
        iterCnt := iterCnt + 1.U
      }
    }
  }

  // Status signals
  io.status.idle    := (state === idle)
  io.status.running := (state === read_and_convert)
}
