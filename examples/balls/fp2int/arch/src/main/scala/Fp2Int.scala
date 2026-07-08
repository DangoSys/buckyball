package examples.balls.fp2int

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class Fp2Int(val b: GlobalConfig) extends Module {
  val elemsPerWord = 4
  val bankWidth    = b.memDomain.bankWidth

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "Fp2IntBall")
    .getOrElse(throw new IllegalArgumentException("Fp2IntBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  require(inBW >= 1, "Fp2Int requires at least one read port")
  require(outBW >= 1, "Fp2Int requires at least one write port")

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val idle :: sReadReq :: sReadResp :: sWriteReq :: sWriteResp :: complete :: Nil = Enum(6)
  val state                                                                       = RegInit(idle)

  val robIdReg    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSubReg    = RegInit(false.B)
  val subRobIdReg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  val rbankReg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbankReg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iterReg  = RegInit(0.U(b.frontend.iter_len.W))
  val scaleReg = RegInit(0.U(32.W))
  val int8Mode = RegInit(false.B)

  val rowReg    = RegInit(0.U(b.frontend.iter_len.W))
  val groupReg  = RegInit(0.U(2.W))
  val outWord   = RegInit(0.U(bankWidth.W))
  val writeWord = RegInit(0.U(bankWidth.W))

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id           := robIdReg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := rbankReg
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id           := robIdReg
    io.bankWrite(i).ball_id          := 0.U
    io.bankWrite(i).bank_id          := wbankReg
    io.bankWrite(i).group_id         := 0.U
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready    := false.B
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg

  def fp32Multiply(a: UInt, bv: UInt): UInt = {
    val aSign     = a(31)
    val bSign     = bv(31)
    val aExp      = a(30, 23)
    val bExp      = bv(30, 23)
    val aMant     = Cat(1.U(1.W), a(22, 0))
    val bMant     = Cat(1.U(1.W), bv(22, 0))
    val resSign   = aSign ^ bSign
    val aZero     = aExp === 0.U && a(22, 0) === 0.U
    val bZero     = bExp === 0.U && bv(22, 0) === 0.U
    val prod      = (aMant * bMant)(47, 0)
    val mant      = Wire(UInt(24.W))
    val expAdjust = Wire(UInt(1.W))
    when(prod(47)) {
      mant      := prod(47, 24)
      expAdjust := 1.U
    }.otherwise {
      mant      := prod(46, 23)
      expAdjust := 0.U
    }
    val expWide   = aExp +& bExp +& expAdjust - 127.U
    val result    = Wire(UInt(32.W))
    when(aZero || bZero) {
      result := 0.U
    }.elsewhen(expWide(9, 8) =/= 0.U && expWide(9)) {
      result := 0.U
    }.elsewhen(expWide(8) && !expWide(9)) {
      result := Cat(resSign, 255.U(8.W), 0.U(23.W))
    }.otherwise {
      result := Cat(resSign, expWide(7, 0), mant(22, 0))
    }
    result
  }

  def fp32ToInt32(fp: UInt): UInt = {
    val sign     = fp(31)
    val exponent = fp(30, 23)
    val mantissa = Cat(1.U(1.W), fp(22, 0))
    val isZero   = exponent === 0.U && fp(22, 0) === 0.U
    val expVal   = exponent.asSInt - 127.S
    val result   = Wire(SInt(32.W))
    when(isZero) {
      result := 0.S
    }.elsewhen(expVal >= 31.S) {
      result := Mux(sign.asBool, -2147483648L.S(32.W), 2147483647.S(32.W))
    }.elsewhen(expVal < 0.S) {
      when(expVal === -1.S) {
        result := Mux(sign.asBool, -1.S(32.W), 1.S(32.W))
      }.otherwise {
        result := 0.S
      }
    }.otherwise {
      val shift = expVal.asUInt(4, 0)
      val mag   = Wire(UInt(32.W))
      when(shift >= 23.U) {
        mag := mantissa << (shift - 23.U)
      }.otherwise {
        mag := mantissa >> (23.U - shift)
      }
      result := Mux(sign.asBool, -(mag.asSInt), mag.asSInt)
    }
    result.asUInt
  }

  def fp32ToInt8(fp: UInt): UInt = {
    val v = fp32ToInt32(fp).asSInt
    val c = Wire(SInt(8.W))
    when(v > 127.S) {
      c := 127.S(8.W)
    }.elsewhen(v < -128.S) {
      c := -128.S(8.W)
    }.otherwise {
      c := v(7, 0).asSInt
    }
    c.asUInt
  }

  def quantWord(data: UInt): (UInt, UInt) = {
    val i32 = Wire(Vec(elemsPerWord, UInt(32.W)))
    val i8  = Wire(Vec(elemsPerWord, UInt(8.W)))
    for (i <- 0 until elemsPerWord) {
      val fp     = data((i + 1) * 32 - 1, i * 32)
      val scaled = fp32Multiply(fp, scaleReg)
      i32(i) := fp32ToInt32(scaled)
      i8(i)  := fp32ToInt8(scaled)
    }
    (Cat(i32.reverse), Cat(i8.reverse))
  }

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val srcCol = io.cmdReq.bits.cmd.op1_col
        val dstCol = io.cmdReq.bits.cmd.wr_col
        val isI32  = srcCol === 1.U && dstCol === 1.U
        val isI8   = srcCol === 4.U && dstCol === 1.U

        assert(io.cmdReq.bits.cmd.iter > 0.U, "Fp2Int iter must be > 0")
        assert(isI32 || isI8, "Fp2Int unsupported bank layout")

        robIdReg    := io.cmdReq.bits.rob_id
        isSubReg    := io.cmdReq.bits.is_sub
        subRobIdReg := io.cmdReq.bits.sub_rob_id
        rbankReg    := io.cmdReq.bits.cmd.op1_bank
        wbankReg    := io.cmdReq.bits.cmd.wr_bank
        iterReg     := io.cmdReq.bits.cmd.iter
        scaleReg    := io.cmdReq.bits.cmd.special(31, 0)
        int8Mode    := isI8
        rowReg      := 0.U
        groupReg    := 0.U
        outWord     := 0.U
        writeWord   := 0.U
        state       := sReadReq
      }
    }

    is(sReadReq) {
      io.bankRead(0).bank_id          := rbankReg
      io.bankRead(0).group_id         := Mux(int8Mode, groupReg, 0.U)
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := rowReg
      when(io.bankRead(0).io.req.fire) {
        state := sReadResp
      }
    }

    is(sReadResp) {
      io.bankRead(0).bank_id       := rbankReg
      io.bankRead(0).group_id      := Mux(int8Mode, groupReg, 0.U)
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        val (i32Word, i8Bytes) = quantWord(io.bankRead(0).io.resp.bits.data)
        when(int8Mode) {
          val nextWord = outWord | (i8Bytes << (groupReg << 5.U))
          when(groupReg === 3.U) {
            writeWord := nextWord
            groupReg  := 0.U
            outWord   := 0.U
            state     := sWriteReq
          }.otherwise {
            outWord  := nextWord
            groupReg := groupReg + 1.U
            state    := sReadReq
          }
        }.otherwise {
          writeWord := i32Word
          state     := sWriteReq
        }
      }
    }

    is(sWriteReq) {
      io.bankWrite(0).bank_id          := wbankReg
      io.bankWrite(0).group_id         := 0.U
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := rowReg
      io.bankWrite(0).io.req.bits.data := writeWord
      io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
      when(io.bankWrite(0).io.req.fire) {
        state := sWriteResp
      }
    }

    is(sWriteResp) {
      io.bankWrite(0).bank_id       := wbankReg
      io.bankWrite(0).group_id      := 0.U
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(rowReg === iterReg - 1.U) {
          state := complete
        }.otherwise {
          rowReg := rowReg + 1.U
          state  := sReadReq
        }
      }
    }

    is(complete) {
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := state === idle
  io.status.running := state =/= idle && state =/= complete
}
