package framework.balldomain.prototype.quant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.quant.configs.QuantBallParam

/**
 * Quant - Quantization core logic.
 *
 * INT32 mode: FP32 -> INT32
 *   Each 128-bit SRAM word = 4 x FP32.
 *   Output: 4 x INT32 = 128 bits. 1:1 read/write.
 *
 * INT8 mode: FP32 -> INT8
 *   Read 4 words (16 FP32), pack into 1 output word (16 x INT8 = 128 bits). 4:1 read/write.
 *
 * Scale factor from cmd.special(31,0) as FP32 bit pattern.
 *
 * FSM follows ReluBall pattern:
 *   idle -> sRead (pipelined read all words, process on resp) -> sWrite (write all results) -> complete -> idle
 */
@instantiable
class Quant(val b: GlobalConfig) extends Module {
  val ballConfig   = QuantBallParam()
  val isInt8       = ballConfig.targetType == "INT8"
  val elemsPerWord = 4
  val bankWidth    = b.memDomain.bankWidth
  // For INT32: InputNum = iter (max 16), for INT8: InputNum = iter (output words, max 16)
  val InputNum     = 16 // max elements per dimension, matching ReluBall

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "QuantBall")
    .getOrElse(throw new IllegalArgumentException("QuantBall not found in config"))
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

  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  when(io.cmdReq.fire) {
    rob_id_reg     := io.cmdReq.bits.rob_id
    is_sub_reg     := io.cmdReq.bits.is_sub
    sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  // FSM
  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state                                      = RegInit(idle)

  // Result storage: up to InputNum output words, each 128 bits
  val regArray = RegInit(VecInit(Seq.fill(InputNum)(0.U(bankWidth.W))))

  val readCounter  = RegInit(0.U(log2Ceil(InputNum * 4 + 1).W)) // request counter
  val respCounter  = RegInit(0.U(log2Ceil(InputNum * 4 + 1).W)) // response counter
  val writeCounter = RegInit(0.U(log2Ceil(InputNum + 1).W))

  val raddr_reg = RegInit(0.U(b.frontend.iter_len.W))
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val waddr_reg = RegInit(0.U(b.frontend.iter_len.W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg  = RegInit(0.U(b.frontend.iter_len.W))
  val scale_reg = RegInit(0.U(32.W))

  // For INT8: accumulate 4 responses into one output word
  val int8AccumIdx = RegInit(0.U(2.W))                      // 0..3 within current output word
  val int8OutIdx   = RegInit(0.U(log2Ceil(InputNum + 1).W)) // output word index

  // Total read requests needed
  val totalReads = Wire(UInt(b.frontend.iter_len.W))
  if (isInt8) {
    totalReads := iter_reg << 2
  } else {
    totalReads := iter_reg
  }

  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(b.memDomain.bankMaskLen, UInt(1.W)))

  // Default outputs
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id         := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
    io.bankWrite(i).bank_id           := wbank_reg
    io.bankWrite(i).group_id          := 0.U
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  // ---- FP32 multiply ----
  def fp32Multiply(a: UInt, bv: UInt): UInt = {
    val a_sign          = a(31)
    val b_sign          = bv(31)
    val a_exp           = a(30, 23)
    val b_exp           = bv(30, 23)
    val a_mant          = Cat(1.U(1.W), a(22, 0))
    val b_mant          = Cat(1.U(1.W), bv(22, 0))
    val result_sign     = a_sign ^ b_sign
    val a_is_zero       = a_exp === 0.U && a(22, 0) === 0.U
    val b_is_zero       = b_exp === 0.U && bv(22, 0) === 0.U
    val mant_product    = (a_mant * b_mant)(47, 0)
    val mant_shifted    = Wire(UInt(24.W))
    val exp_adjust      = Wire(UInt(1.W))
    when(mant_product(47)) {
      mant_shifted := mant_product(47, 24)
      exp_adjust   := 1.U
    }.otherwise {
      mant_shifted := mant_product(46, 23)
      exp_adjust   := 0.U
    }
    val result_exp_wide = a_exp +& b_exp +& exp_adjust - 127.U
    val result_exp      = result_exp_wide(7, 0)
    val result          = Wire(UInt(32.W))
    when(a_is_zero || b_is_zero) {
      result := 0.U
    }.elsewhen(result_exp_wide(9, 8) =/= 0.U && result_exp_wide(9)) {
      result := 0.U
    }.elsewhen(result_exp_wide(8) && !result_exp_wide(9)) {
      result := Cat(result_sign, 255.U(8.W), 0.U(23.W))
    }.otherwise {
      result := Cat(result_sign, result_exp, mant_shifted(22, 0))
    }
    result
  }

  def fp32ToInt32(fp: UInt): UInt = {
    val sign     = fp(31)
    val exponent = fp(30, 23)
    val mantissa = Cat(1.U(1.W), fp(22, 0))
    val is_zero  = exponent === 0.U && fp(22, 0) === 0.U
    val exp_val  = exponent.asSInt - 127.S
    val result   = Wire(SInt(32.W))
    when(is_zero) {
      result := 0.S
    }.elsewhen(exp_val >= 31.S) {
      result := Mux(sign.asBool, -2147483648L.S(32.W), 2147483647.S(32.W))
    }.elsewhen(exp_val < 0.S) {
      when(exp_val === -1.S) {
        result := Mux(sign.asBool, -1.S(32.W), 1.S(32.W))
      }.otherwise {
        result := 0.S
      }
    }.otherwise {
      val shift_amount = exp_val.asUInt(4, 0)
      val magnitude    = Wire(UInt(32.W))
      when(shift_amount >= 23.U) {
        magnitude := mantissa << (shift_amount - 23.U)
      }.otherwise {
        magnitude := mantissa >> (23.U - shift_amount)
      }
      result := Mux(sign.asBool, -(magnitude.asSInt), magnitude.asSInt)
    }
    result.asUInt
  }

  def fp32ToInt8(fp: UInt): UInt = {
    val int32val = fp32ToInt32(fp).asSInt
    val clamped  = Wire(SInt(8.W))
    when(int32val > 127.S) {
      clamped := 127.S(8.W)
    }.elsewhen(int32val < -128.S) {
      clamped := -128.S(8.W)
    }.otherwise {
      clamped := int32val(7, 0).asSInt
    }
    clamped.asUInt
  }

  // ---- FSM ----
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state        := sRead
        readCounter  := 0.U
        respCounter  := 0.U
        writeCounter := 0.U
        raddr_reg    := 0.U
        rbank_reg    := io.cmdReq.bits.cmd.op1_bank
        waddr_reg    := 0.U
        wbank_reg    := io.cmdReq.bits.cmd.wr_bank
        iter_reg     := io.cmdReq.bits.cmd.iter
        scale_reg    := io.cmdReq.bits.cmd.special(31, 0)
        if (isInt8) {
          int8AccumIdx := 0.U
          int8OutIdx   := 0.U
        }
      }
    }

    is(sRead) {
      io.bankRead(0).io.resp.ready := true.B

      // Send read requests (pipelined)
      io.bankRead(0).io.req.valid     := readCounter < totalReads
      io.bankRead(0).io.req.bits.addr := raddr_reg + readCounter

      when(io.bankRead(0).io.req.fire) {
        readCounter := readCounter + 1.U
      }

      // Process responses (1 cycle latency after req)
      val dataWord = io.bankRead(0).io.resp.bits.data

      when(io.bankRead(0).io.resp.fire) {
        if (isInt8) {
          // Quantize 4 FP32 -> 4 INT8, store in current output word
          val packedBytes = Wire(Vec(4, UInt(8.W)))
          for (i <- 0 until elemsPerWord) {
            val fp_elem = dataWord((i + 1) * 32 - 1, i * 32)
            val scaled  = fp32Multiply(fp_elem, scale_reg)
            packedBytes(i) := fp32ToInt8(scaled)
          }
          // Write 4 bytes into the correct position of regArray(int8OutIdx)
          // Position: int8AccumIdx * 32 bits
          val shift = int8AccumIdx * 32.U
          val mask    = ~(Fill(32, 1.U(1.W)) << shift)
          val newBits = Cat(packedBytes(3), packedBytes(2), packedBytes(1), packedBytes(0))
          regArray(int8OutIdx) := (regArray(int8OutIdx) & mask) | (newBits.asUInt << shift)

          when(int8AccumIdx === 3.U) {
            int8AccumIdx := 0.U
            int8OutIdx   := int8OutIdx + 1.U
          }.otherwise {
            int8AccumIdx := int8AccumIdx + 1.U
          }
        } else {
          // INT32 mode: quantize 4 FP32 -> 4 INT32, pack into 128-bit word
          val results = Wire(Vec(elemsPerWord, UInt(32.W)))
          for (i <- 0 until elemsPerWord) {
            val fp_elem = dataWord((i + 1) * 32 - 1, i * 32)
            val scaled  = fp32Multiply(fp_elem, scale_reg)
            results(i) := fp32ToInt32(scaled)
          }
          regArray(respCounter) := Cat(results.reverse)
        }
        respCounter := respCounter + 1.U

        // Check if all responses received
        val totalResps = if (isInt8) iter_reg << 2 else iter_reg
        when(respCounter === (totalResps - 1.U)) {
          state := sWrite
        }
      }
    }

    is(sWrite) {
      val hasMore = writeCounter < iter_reg

      io.bankWrite(0).io.req.valid     := hasMore
      io.bankWrite(0).io.req.bits.addr := waddr_reg + writeCounter
      io.bankWrite(0).io.req.bits.data := regArray(writeCounter)
      io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))
      io.bankWrite(0).io.resp.ready    := true.B

      when(io.bankWrite(0).io.req.fire) {
        when(writeCounter === (iter_reg - 1.U)) {
          state := complete
        }.otherwise {
          writeCounter := writeCounter + 1.U
        }
      }
    }

    is(complete) {
      io.bankWrite(0).io.resp.ready := true.B
      io.cmdResp.valid              := true.B
      io.cmdResp.bits.rob_id        := rob_id_reg
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := state === idle
  io.status.running := (state === sRead) || (state === sWrite)
}
