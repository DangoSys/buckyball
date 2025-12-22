package prototype.abft

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.experimental.{SerializableModule, SerializableModuleParameter}
import org.chipsalliance.cde.config.Parameters
import framework.memdomain.backend.banks.{SramReadIO, SramWriteIO}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.Status
import prototype.abft.configs.ABFTConfig

/**
 * ABFTSystolicArray - Simple systolic array with ABFT (Algorithm-Based Fault Tolerance)
 *
 * ABFT mechanism:
 * - Matrix A has an extra checksum row (sum of each column)
 * - Matrix B has an extra checksum column (sum of each row)
 * - Result matrix C will have checksum row and column
 * - Verify checksums match to detect errors
 *
 * Simple implementation: process InputNum x InputNum tiles
 */
@instantiable
class ABFTSystolicArray(val parameter: ABFTConfig)(implicit p: Parameters)
    extends Module
    with SerializableModule[ABFTConfig] {
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

    // Unified bank write interface for accumulator writes (accumulate mode)
    val bankWriteAcc =
      Vec(ballParam.numBanks, Flipped(new SramWriteIO(ballParam.bankEntries, accWidth, ballParam.bankMaskLen)))

    // Status output
    val status = new Status
  })

  // State machine
  val idle :: sLoadA :: sLoadB :: sCompute :: sWrite :: sCheck :: complete :: Nil = Enum(7)
  val state                                                                       = RegInit(idle)

  // Registers for matrix A (InputNum x InputNum)
  val matrixA = RegInit(
    VecInit(Seq.fill(InputNum)(
      VecInit(Seq.fill(InputNum)(0.S(inputWidth.W)))
    ))
  )

  // Registers for matrix B (InputNum x InputNum)
  val matrixB = RegInit(
    VecInit(Seq.fill(InputNum)(
      VecInit(Seq.fill(InputNum)(0.S(inputWidth.W)))
    ))
  )

  // Result matrix C (InputNum x InputNum)
  val matrixC = RegInit(
    VecInit(Seq.fill(InputNum)(
      VecInit(Seq.fill(InputNum)(0.S(32.W)))
    ))
  )

  // Checksum registers
  val checksumA_row = RegInit(VecInit(Seq.fill(InputNum)(0.S(32.W))))
  val checksumB_col = RegInit(VecInit(Seq.fill(InputNum)(0.S(32.W))))
  val checksumC_row = RegInit(VecInit(Seq.fill(InputNum)(0.S(32.W))))
  val checksumC_col = RegInit(VecInit(Seq.fill(InputNum)(0.S(32.W))))

  // Counters
  val rowCounter   = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val colCounter   = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val readCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(InputNum + 1).W))

  // Instruction registers
  val robid_reg    = RegInit(0.U(10.W))
  val op1_addr_reg = RegInit(0.U(10.W))
  val op1_bank_reg = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val op2_addr_reg = RegInit(0.U(10.W))
  val op2_bank_reg = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val wr_addr_reg  = RegInit(0.U(10.W))
  val wr_bank_reg  = RegInit(0.U(log2Up(ballParam.numBanks).W))
  val iter_reg     = RegInit(0.U(10.W))
  val cycle_reg    = RegInit(0.U(6.W))
  val iterCnt      = RegInit(0.U(32.W))

  // Error detection flag
  val errorDetected = RegInit(false.B)

  // Write data register
  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(ballParam.bankMaskLen, UInt(1.W)))

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

  // State machine
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state         := sLoadA
        readCounter   := 0.U
        rowCounter    := 0.U
        colCounter    := 0.U
        writeCounter  := 0.U
        errorDetected := false.B

        robid_reg    := io.cmdReq.bits.rob_id
        op1_addr_reg := 0.U // New ISA: all operations start from row 0
        op1_bank_reg := io.cmdReq.bits.cmd.op1_bank
        op2_addr_reg := 0.U // New ISA: all operations start from row 0
        op2_bank_reg := io.cmdReq.bits.cmd.op2_bank
        wr_addr_reg  := 0.U // New ISA: all operations start from row 0
        wr_bank_reg  := io.cmdReq.bits.cmd.wr_bank
        iter_reg     := io.cmdReq.bits.cmd.iter
        cycle_reg    := (io.cmdReq.bits.cmd.iter +& (InputNum.U - 1.U)) / InputNum.U - 1.U
      }
    }

    is(sLoadA) {
      // Load matrix A row by row
      when(readCounter < InputNum.U) {
        io.bankRead(op1_bank_reg).req.valid     := true.B
        io.bankRead(op1_bank_reg).req.bits.addr := op1_addr_reg + readCounter
        readCounter                             := readCounter + 1.U
      }

      io.bankRead(op1_bank_reg).resp.ready := true.B
      when(io.bankRead(op1_bank_reg).resp.fire) {
        for (col <- 0 until InputNum) {
          val hi  = (col + 1) * inputWidth - 1
          val lo  = col * inputWidth
          val raw = io.bankRead(op1_bank_reg).resp.bits.data(hi, lo)
          matrixA(rowCounter)(col) := raw.asSInt
        }
        rowCounter := rowCounter + 1.U
      }

      when(rowCounter === InputNum.U) {
        state       := sLoadB
        readCounter := 0.U
        rowCounter  := 0.U
        // Compute checksum for matrix A (sum of each column)
        for (col <- 0 until InputNum) {
          checksumA_row(col) := (0 until InputNum).map(i => matrixA(i)(col)).reduce(_ + _)
        }
      }
    }

    is(sLoadB) {
      // Load matrix B row by row (same as matrix A)
      when(readCounter < InputNum.U) {
        io.bankRead(op2_bank_reg).req.valid     := true.B
        io.bankRead(op2_bank_reg).req.bits.addr := op2_addr_reg + readCounter
        readCounter                             := readCounter + 1.U
      }

      io.bankRead(op2_bank_reg).resp.ready := true.B
      when(io.bankRead(op2_bank_reg).resp.fire) {
        for (col <- 0 until InputNum) {
          val hi  = (col + 1) * inputWidth - 1
          val lo  = col * inputWidth
          val raw = io.bankRead(op2_bank_reg).resp.bits.data(hi, lo)
          matrixB(rowCounter)(col) := raw.asSInt
        }
        rowCounter := rowCounter + 1.U
      }

      when(rowCounter === InputNum.U) {
        state      := sCompute
        rowCounter := 0.U
        colCounter := 0.U
        // Compute checksum for matrix B (sum of each row)
        for (row <- 0 until InputNum) {
          checksumB_col(row) := (0 until InputNum).map(j => matrixB(row)(j)).reduce(_ + _)
        }
      }
    }

    is(sCompute) {
      // Simple systolic array computation: C[i][j] = sum(A[i][k] * B[k][j])
      // Compute all elements in one cycle (simple implementation)
      for (i <- 0 until InputNum) {
        for (j <- 0 until InputNum) {
          val sum = (0 until InputNum).map(k => matrixA(i)(k) * matrixB(k)(j)).reduce((a: SInt, b: SInt) => a + b)
          matrixC(i)(j) := sum
        }
      }

      // Compute checksums for result matrix C
      for (col <- 0 until InputNum) {
        checksumC_row(col) := (0 until InputNum).map(i => matrixC(i)(col)).reduce((a: SInt, b: SInt) => a + b)
      }
      for (row <- 0 until InputNum) {
        checksumC_col(row) := (0 until InputNum).map(j => matrixC(row)(j)).reduce((a: SInt, b: SInt) => a + b)
      }

      state := sCheck
    }

    is(sCheck) {
      // ABFT verification: check if checksums match
      // For C = A * B, checksum of row i in C should equal sum(A[i][k] * checksumB_col[k])
      // For C = A * B, checksum of col j in C should equal sum(checksumA_row[k] * B[k][j])
      // Simple check: verify first row and first column checksums
      val expectedRowChecksum =
        (0 until InputNum).map(k => matrixA(0)(k) * checksumB_col(k)).reduce((a: SInt, b: SInt) => a + b)

      val expectedColChecksum =
        (0 until InputNum).map(k => checksumA_row(k) * matrixB(k)(0)).reduce((a: SInt, b: SInt) => a + b)

      val rowMatch            = checksumC_row(0) === expectedRowChecksum
      val colMatch            = checksumC_col(0) === expectedColChecksum

      errorDetected := !rowMatch || !colMatch
      state         := sWrite
      writeCounter  := 0.U
      // Prepare first write data (clamp 32-bit result to 8-bit)
      writeDataReg  := Cat((0 until InputNum).reverse.map { j =>
        val clamped =
          Mux(matrixC(0)(j) > 127.S, 127.U, Mux(matrixC(0)(j) < -128.S, -128.S.asUInt, matrixC(0)(j)(inputWidth - 1, 0)))
        clamped
      })
      for (i <- 0 until parameter.bankMaskLen) {
        writeMaskReg(i) := 1.U(1.W)
      }
    }

    is(sWrite) {
      // Write results back to scratchpad
      io.bankWrite(wr_bank_reg).req.valid     := writeCounter < InputNum.U
      io.bankWrite(wr_bank_reg).req.bits.addr := wr_addr_reg + writeCounter
      io.bankWrite(wr_bank_reg).req.bits.data := writeDataReg
      io.bankWrite(wr_bank_reg).req.bits.mask := writeMaskReg

      when(io.bankWrite(wr_bank_reg).req.fire) {
        when(writeCounter === (InputNum - 1).U) {
          state := complete
        }.otherwise {
          writeCounter := writeCounter + 1.U
          // Prepare next row's write data (clamp 32-bit result to 8-bit)
          val nextRow = writeCounter + 1.U
          writeDataReg := Cat((0 until InputNum).reverse.map { j =>
            val idx     = nextRow
            val clamped = Mux(
              matrixC(idx)(j) > 127.S,
              127.U,
              Mux(matrixC(idx)(j) < -128.S, -128.S.asUInt, matrixC(idx)(j)(inputWidth - 1, 0))
            )
            clamped
          })
        }
      }
    }

    is(complete) {
      when(cycle_reg === 0.U) {
        io.cmdResp.valid       := true.B
        io.cmdResp.bits.rob_id := robid_reg
        when(io.cmdResp.fire) {
          iterCnt := iterCnt + 1.U
        }
      }
      state := idle
    }
  }

  // Status signals
  io.status.ready    := io.cmdReq.ready
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := (state === idle)
  io.status.init     := (state === sLoadA) || (state === sLoadB)
  io.status.running  := (state === sCompute) || (state === sCheck) || (state === sWrite)
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter     := iterCnt

  // Reset handling
  when(reset.asBool) {
    for (i <- 0 until InputNum) {
      for (j <- 0 until InputNum) {
        matrixA(i)(j) := 0.S
        matrixB(i)(j) := 0.S
        matrixC(i)(j) := 0.S
      }
      checksumA_row(i) := 0.S
      checksumB_col(i) := 0.S
      checksumC_row(i) := 0.S
      checksumC_col(i) := 0.S
    }
    writeDataReg := 0.U
    for (i <- 0 until parameter.bankMaskLen) {
      writeMaskReg(i) := 0.U
    }
    errorDetected := false.B
  }
}
