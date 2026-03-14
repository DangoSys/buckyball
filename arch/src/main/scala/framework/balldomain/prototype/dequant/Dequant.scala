package framework.balldomain.prototype.dequant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

/**
 * Dequant - Dequantization core logic.
 * INT32 -> FP32: fp32_val = int32_val * scale
 * Each 128-bit SRAM word = 4 x INT32. Output: 4 x FP32 = 128 bits. 1:1 read/write.
 * Scale from cmd.special(31,0) as FP32 bit pattern.
 *
 * FSM follows ReluBall pattern:
 *   idle -> sRead -> sWrite -> complete -> idle
 */
@instantiable
class Dequant(val b: GlobalConfig) extends Module {
  val elemsPerWord = 4
  val bankWidth    = b.memDomain.bankWidth
  val InputNum     = 16

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "DequantBall")
    .getOrElse(throw new IllegalArgumentException("DequantBall not found in config"))
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

  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state                                      = RegInit(idle)

  val regArray = RegInit(VecInit(Seq.fill(InputNum)(0.U(bankWidth.W))))

  val readCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val respCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(InputNum + 1).W))

  val raddr_reg = RegInit(0.U(b.frontend.iter_len.W))
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val waddr_reg = RegInit(0.U(b.frontend.iter_len.W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg  = RegInit(0.U(b.frontend.iter_len.W))
  val scale_reg = RegInit(0.U(32.W))

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

  // INT32 to FP32
  def int32ToFp32(intVal: UInt): UInt = {
    val signed  = intVal.asSInt
    val is_zero = signed === 0.S
    val sign    = intVal(31)
    val absVal  = Wire(UInt(32.W))
    absVal := Mux(sign.asBool, (~intVal + 1.U), intVal)

    val leadingOne = Wire(UInt(5.W))
    leadingOne := (30.U - PriorityEncoder(Reverse(absVal(30, 0))))

    val exponent = Wire(UInt(8.W))
    exponent := leadingOne +& 127.U

    val mantissa = Wire(UInt(23.W))
    when(leadingOne >= 23.U) {
      mantissa := (absVal >> (leadingOne - 23.U))(22, 0)
    }.otherwise {
      mantissa := (absVal << (23.U - leadingOne))(22, 0)
    }

    val result = Wire(UInt(32.W))
    when(is_zero) {
      result := 0.U
    }.otherwise {
      result := Cat(sign, exponent, mantissa)
    }
    result
  }

  // FP32 multiply
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

  // FSM
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
      }
    }

    is(sRead) {
      io.bankRead(0).io.resp.ready := true.B

      io.bankRead(0).io.req.valid     := readCounter < iter_reg
      io.bankRead(0).io.req.bits.addr := raddr_reg + readCounter

      when(io.bankRead(0).io.req.fire) {
        readCounter := readCounter + 1.U
      }

      val dataWord = io.bankRead(0).io.resp.bits.data

      when(io.bankRead(0).io.resp.fire) {
        val results = Wire(Vec(elemsPerWord, UInt(32.W)))
        for (i <- 0 until elemsPerWord) {
          val int_elem = dataWord((i + 1) * 32 - 1, i * 32)
          val fp_elem  = int32ToFp32(int_elem)
          results(i) := fp32Multiply(fp_elem, scale_reg)
        }
        regArray(respCounter) := Cat(results.reverse)
        respCounter := respCounter + 1.U

        when(respCounter === (iter_reg - 1.U)) {
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
