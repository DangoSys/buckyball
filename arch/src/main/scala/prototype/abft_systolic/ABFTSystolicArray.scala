package prototype.abft_systolic

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}
import framework.builtin.frontend.rs.{BallRsIssue, BallRsComplete}
import examples.BuckyballConfigs.CustomBuckyballConfig
import framework.blink.Status

/**
 * ABFTSystolicArray - Simple systolic array with ABFT (Algorithm-Based Fault Tolerance)
 *
 * ABFT mechanism:
 * - Matrix A has an extra checksum row (sum of each column)
 * - Matrix B has an extra checksum column (sum of each row)
 * - Result matrix C will have checksum row and column
 * - Verify checksums match to detect errors
 *
 * Simple implementation: process veclane x veclane tiles
 */
class ABFTSystolicArray(implicit b: CustomBuckyballConfig, p: Parameters) extends Module {
  val spad_w = b.veclane * b.inputType.getWidth

  val io = IO(new Bundle {
    // Command interface
    val cmdReq = Flipped(Decoupled(new BallRsIssue))
    val cmdResp = Decoupled(new BallRsComplete)

    // Scratchpad SRAM read/write interface
    val sramRead = Vec(b.sp_banks, Flipped(new SramReadIO(b.spad_bank_entries, spad_w)))
    val sramWrite = Vec(b.sp_banks, Flipped(new SramWriteIO(b.spad_bank_entries, spad_w, b.spad_mask_len)))

    // Accumulator write interface (for partial sums)
    val accWrite = Vec(b.acc_banks, Flipped(new SramWriteIO(b.acc_bank_entries, b.acc_w, b.acc_mask_len)))

    // Status output
    val status = new Status
  })

  // State machine
  val idle :: sLoadA :: sLoadB :: sCompute :: sWrite :: sCheck :: complete :: Nil = Enum(7)
  val state = RegInit(idle)

  // Registers for matrix A (veclane x veclane)
  val matrixA = RegInit(
    VecInit(Seq.fill(b.veclane)(
      VecInit(Seq.fill(b.veclane)(0.S(b.inputType.getWidth.W)))
    ))
  )

  // Registers for matrix B (veclane x veclane)
  val matrixB = RegInit(
    VecInit(Seq.fill(b.veclane)(
      VecInit(Seq.fill(b.veclane)(0.S(b.inputType.getWidth.W)))
    ))
  )

  // Result matrix C (veclane x veclane)
  val matrixC = RegInit(
    VecInit(Seq.fill(b.veclane)(
      VecInit(Seq.fill(b.veclane)(0.S(32.W)))
    ))
  )

  // Checksum registers
  val checksumA_row = RegInit(VecInit(Seq.fill(b.veclane)(0.S(32.W))))
  val checksumB_col = RegInit(VecInit(Seq.fill(b.veclane)(0.S(32.W))))
  val checksumC_row = RegInit(VecInit(Seq.fill(b.veclane)(0.S(32.W))))
  val checksumC_col = RegInit(VecInit(Seq.fill(b.veclane)(0.S(32.W))))

  // Counters
  val rowCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val colCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val readCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(b.veclane + 1).W))

  // Instruction registers
  val robid_reg = RegInit(0.U(10.W))
  val op1_addr_reg = RegInit(0.U(10.W))
  val op1_bank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val op2_addr_reg = RegInit(0.U(10.W))
  val op2_bank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val wr_addr_reg = RegInit(0.U(10.W))
  val wr_bank_reg = RegInit(0.U(log2Up(b.sp_banks).W))
  val iter_reg = RegInit(0.U(10.W))
  val cycle_reg = RegInit(0.U(6.W))
  val iterCnt = RegInit(0.U(32.W))

  // Error detection flag
  val errorDetected = RegInit(false.B)

  // Write data register
  val writeDataReg = Reg(UInt(spad_w.W))
  val writeMaskReg = Reg(Vec(b.spad_mask_len, UInt(1.W)))

  // Default SRAM assignments
  for (i <- 0 until b.sp_banks) {
    io.sramRead(i).req.valid := false.B
    io.sramRead(i).req.bits.addr := 0.U
    io.sramRead(i).req.bits.fromDMA := false.B
    io.sramRead(i).resp.ready := false.B

    io.sramWrite(i).req.valid := false.B
    io.sramWrite(i).req.bits.addr := 0.U
    io.sramWrite(i).req.bits.data := 0.U
    io.sramWrite(i).req.bits.mask := VecInit(Seq.fill(b.spad_mask_len)(0.U(1.W)))
  }

  // Default accumulator assignments
  for (i <- 0 until b.acc_banks) {
    io.accWrite(i).req.valid := false.B
    io.accWrite(i).req.bits.addr := 0.U
    io.accWrite(i).req.bits.data := 0.U
    io.accWrite(i).req.bits.mask := VecInit(Seq.fill(b.acc_mask_len)(0.U(1.W)))
  }

  // Command interface defaults
  io.cmdReq.ready := state === idle
  io.cmdResp.valid := false.B
  io.cmdResp.bits.rob_id := robid_reg

  // State machine
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state := sLoadA
        readCounter := 0.U
        rowCounter := 0.U
        colCounter := 0.U
        writeCounter := 0.U
        errorDetected := false.B

        robid_reg := io.cmdReq.bits.rob_id
        op1_addr_reg := io.cmdReq.bits.cmd.op1_bank_addr
        op1_bank_reg := io.cmdReq.bits.cmd.op1_bank
        op2_addr_reg := io.cmdReq.bits.cmd.op2_bank_addr
        op2_bank_reg := io.cmdReq.bits.cmd.op2_bank
        wr_addr_reg := io.cmdReq.bits.cmd.wr_bank_addr
        wr_bank_reg := io.cmdReq.bits.cmd.wr_bank
        iter_reg := io.cmdReq.bits.cmd.iter
        cycle_reg := (io.cmdReq.bits.cmd.iter +& (b.veclane.U - 1.U)) / b.veclane.U - 1.U
      }
    }

    is(sLoadA) {
      // Load matrix A row by row
      when(readCounter < b.veclane.U) {
        io.sramRead(op1_bank_reg).req.valid := true.B
        io.sramRead(op1_bank_reg).req.bits.addr := op1_addr_reg + readCounter
        readCounter := readCounter + 1.U
      }

      io.sramRead(op1_bank_reg).resp.ready := true.B
      when(io.sramRead(op1_bank_reg).resp.fire) {
        for (col <- 0 until b.veclane) {
          val hi = (col + 1) * b.inputType.getWidth - 1
          val lo = col * b.inputType.getWidth
          val raw = io.sramRead(op1_bank_reg).resp.bits.data(hi, lo)
          matrixA(rowCounter)(col) := raw.asSInt
        }
        rowCounter := rowCounter + 1.U
      }

      when(rowCounter === b.veclane.U) {
        state := sLoadB
        readCounter := 0.U
        rowCounter := 0.U
        // Compute checksum for matrix A (sum of each column)
        for (col <- 0 until b.veclane) {
          checksumA_row(col) := (0 until b.veclane).map(i => matrixA(i)(col)).reduce(_ + _)
        }
      }
    }

    is(sLoadB) {
      // Load matrix B row by row (same as matrix A)
      when(readCounter < b.veclane.U) {
        io.sramRead(op2_bank_reg).req.valid := true.B
        io.sramRead(op2_bank_reg).req.bits.addr := op2_addr_reg + readCounter
        readCounter := readCounter + 1.U
      }

      io.sramRead(op2_bank_reg).resp.ready := true.B
      when(io.sramRead(op2_bank_reg).resp.fire) {
        for (col <- 0 until b.veclane) {
          val hi = (col + 1) * b.inputType.getWidth - 1
          val lo = col * b.inputType.getWidth
          val raw = io.sramRead(op2_bank_reg).resp.bits.data(hi, lo)
          matrixB(rowCounter)(col) := raw.asSInt
        }
        rowCounter := rowCounter + 1.U
      }

      when(rowCounter === b.veclane.U) {
        state := sCompute
        rowCounter := 0.U
        colCounter := 0.U
        // Compute checksum for matrix B (sum of each row)
        for (row <- 0 until b.veclane) {
          checksumB_col(row) := (0 until b.veclane).map(j => matrixB(row)(j)).reduce(_ + _)
        }
      }
    }

    is(sCompute) {
      // Simple systolic array computation: C[i][j] = sum(A[i][k] * B[k][j])
      // Compute all elements in one cycle (simple implementation)
      for (i <- 0 until b.veclane) {
        for (j <- 0 until b.veclane) {
          val sum = (0 until b.veclane).map(k => matrixA(i)(k) * matrixB(k)(j)).reduce(_ + _)
          matrixC(i)(j) := sum
        }
      }

      // Compute checksums for result matrix C
      for (col <- 0 until b.veclane) {
        checksumC_row(col) := (0 until b.veclane).map(i => matrixC(i)(col)).reduce(_ + _)
      }
      for (row <- 0 until b.veclane) {
        checksumC_col(row) := (0 until b.veclane).map(j => matrixC(row)(j)).reduce(_ + _)
      }

      state := sCheck
    }

    is(sCheck) {
      // ABFT verification: check if checksums match
      // For C = A * B, checksum of row i in C should equal sum(A[i][k] * checksumB_col[k])
      // For C = A * B, checksum of col j in C should equal sum(checksumA_row[k] * B[k][j])
      // Simple check: verify first row and first column checksums
      val expectedRowChecksum = (0 until b.veclane).map(k =>
        matrixA(0)(k) * checksumB_col(k)
      ).reduce(_ + _)

      val expectedColChecksum = (0 until b.veclane).map(k =>
        checksumA_row(k) * matrixB(k)(0)
      ).reduce(_ + _)

      val rowMatch = checksumC_row(0) === expectedRowChecksum
      val colMatch = checksumC_col(0) === expectedColChecksum

      errorDetected := !rowMatch || !colMatch
      state := sWrite
      writeCounter := 0.U
      // Prepare first write data (clamp 32-bit result to 8-bit)
      writeDataReg := Cat((0 until b.veclane).reverse.map(j => {
        val clamped = Mux(matrixC(0)(j) > 127.S, 127.U,
                     Mux(matrixC(0)(j) < (-128).S, (-128).S.asUInt,
                     matrixC(0)(j)(b.inputType.getWidth - 1, 0)))
        clamped
      }))
      for (i <- 0 until b.spad_mask_len) {
        writeMaskReg(i) := 1.U(1.W)
      }
    }

    is(sWrite) {
      // Write results back to scratchpad
      io.sramWrite(wr_bank_reg).req.valid := writeCounter < b.veclane.U
      io.sramWrite(wr_bank_reg).req.bits.addr := wr_addr_reg + writeCounter
      io.sramWrite(wr_bank_reg).req.bits.data := writeDataReg
      io.sramWrite(wr_bank_reg).req.bits.mask := writeMaskReg

      when(io.sramWrite(wr_bank_reg).req.fire) {
        when(writeCounter === (b.veclane - 1).U) {
          state := complete
        }.otherwise {
          writeCounter := writeCounter + 1.U
          // Prepare next row's write data (clamp 32-bit result to 8-bit)
          val nextRow = writeCounter + 1.U
          writeDataReg := Cat((0 until b.veclane).reverse.map(j => {
            val idx = nextRow
            val clamped = Mux(matrixC(idx)(j) > 127.S, 127.U,
                         Mux(matrixC(idx)(j) < (-128).S, (-128).S.asUInt,
                         matrixC(idx)(j)(b.inputType.getWidth - 1, 0)))
            clamped
          }))
        }
      }
    }

    is(complete) {
      when(cycle_reg === 0.U) {
        io.cmdResp.valid := true.B
        io.cmdResp.bits.rob_id := robid_reg
        when(io.cmdResp.fire) {
          iterCnt := iterCnt + 1.U
        }
      }
      state := idle
    }
  }

  // Status signals
  io.status.ready := io.cmdReq.ready
  io.status.valid := io.cmdResp.valid
  io.status.idle := (state === idle)
  io.status.init := (state === sLoadA) || (state === sLoadB)
  io.status.running := (state === sCompute) || (state === sCheck) || (state === sWrite)
  io.status.complete := (state === complete) && io.cmdResp.fire
  io.status.iter := iterCnt

  // Reset handling
  when(reset.asBool) {
    for (i <- 0 until b.veclane) {
      for (j <- 0 until b.veclane) {
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
    for (i <- 0 until b.spad_mask_len) {
      writeMaskReg(i) := 0.U
    }
    errorDetected := false.B
  }
}
