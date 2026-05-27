package framework.balldomain.prototype.int2fp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig

@instantiable
class Int2Fp(val b: GlobalConfig) extends Module {
  val elemsPerWord = 4
  val bankWidth    = b.memDomain.bankWidth

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "Int2FpBall")
    .getOrElse(throw new IllegalArgumentException("Int2FpBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  require(inBW >= 1, "Int2Fp requires at least one read port")
  require(outBW >= 1, "Int2Fp requires at least one write port")

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

  val rowReg       = RegInit(0.U(b.frontend.iter_len.W))
  val groupReg     = RegInit(0.U(2.W))
  val modeI8ToFp   = RegInit(false.B)
  val modeI32Group = RegInit(false.B)
  val srcWord      = RegInit(0.U(bankWidth.W))
  val writeWord    = RegInit(0.U(bankWidth.W))

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
    io.bankWrite(i).rob_id            := robIdReg
    io.bankWrite(i).ball_id           := 0.U
    io.bankWrite(i).bank_id           := wbankReg
    io.bankWrite(i).group_id          := 0.U
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg

  def int32ToFp32(intVal: UInt): UInt = {
    val signed = intVal.asSInt
    val isZero = signed === 0.S
    val sign   = intVal(31)
    val absVal = Wire(UInt(32.W))
    absVal := Mux(sign.asBool, ~intVal + 1.U, intVal)

    val leadingOne = Wire(UInt(5.W))
    leadingOne := 30.U - PriorityEncoder(Reverse(absVal(30, 0)))

    val exponent = leadingOne +& 127.U
    val mantissa = Wire(UInt(23.W))
    when(leadingOne >= 23.U) {
      mantissa := (absVal >> (leadingOne - 23.U))(22, 0)
    }.otherwise {
      mantissa := (absVal << (23.U - leadingOne))(22, 0)
    }

    val result = Wire(UInt(32.W))
    when(isZero) {
      result := 0.U
    }.otherwise {
      result := Cat(sign, exponent(7, 0), mantissa)
    }
    result
  }

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

  def scaledI32Word(data: UInt): UInt = {
    val out = Wire(Vec(elemsPerWord, UInt(32.W)))
    for (i <- 0 until elemsPerWord) {
      val elem = data((i + 1) * 32 - 1, i * 32)
      out(i) := fp32Multiply(int32ToFp32(elem), scaleReg)
    }
    Cat(out.reverse)
  }

  def scaledI8Word(data: UInt, group: UInt): UInt = {
    val out = Wire(Vec(elemsPerWord, UInt(32.W)))
    for (i <- 0 until elemsPerWord) {
      val idx  = group * 4.U + i.U
      val byte = (data >> (idx << 3.U))(7, 0)
      val sx   = Cat(Fill(24, byte(7)), byte)
      out(i) := fp32Multiply(int32ToFp32(sx), scaleReg)
    }
    Cat(out.reverse)
  }

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val srcCol      = io.cmdReq.bits.cmd.op1_col
        val dstCol      = io.cmdReq.bits.cmd.wr_col
        val isI32Single = srcCol === 1.U && dstCol === 1.U
        val isI8ToFp    = srcCol === 1.U && dstCol === 4.U
        val isI32Group  = srcCol === 4.U && dstCol === 4.U

        assert(io.cmdReq.bits.cmd.iter > 0.U, "Int2Fp iter must be > 0")
        assert(isI32Single || isI8ToFp || isI32Group, "Int2Fp unsupported bank layout")

        robIdReg     := io.cmdReq.bits.rob_id
        isSubReg     := io.cmdReq.bits.is_sub
        subRobIdReg  := io.cmdReq.bits.sub_rob_id
        rbankReg     := io.cmdReq.bits.cmd.op1_bank
        wbankReg     := io.cmdReq.bits.cmd.wr_bank
        iterReg      := io.cmdReq.bits.cmd.iter
        scaleReg     := io.cmdReq.bits.cmd.special(31, 0)
        rowReg       := 0.U
        groupReg     := 0.U
        modeI8ToFp   := isI8ToFp
        modeI32Group := isI32Group
        srcWord      := 0.U
        writeWord    := 0.U
        state        := sReadReq
      }
    }

    is(sReadReq) {
      io.bankRead(0).bank_id          := rbankReg
      io.bankRead(0).group_id         := Mux(modeI32Group, groupReg, 0.U)
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := rowReg
      when(io.bankRead(0).io.req.fire) {
        state := sReadResp
      }
    }

    is(sReadResp) {
      io.bankRead(0).bank_id       := rbankReg
      io.bankRead(0).group_id      := Mux(modeI32Group, groupReg, 0.U)
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        when(modeI8ToFp) {
          srcWord   := io.bankRead(0).io.resp.bits.data
          writeWord := scaledI8Word(io.bankRead(0).io.resp.bits.data, 0.U)
          groupReg  := 0.U
        }.otherwise {
          writeWord := scaledI32Word(io.bankRead(0).io.resp.bits.data)
        }
        state := sWriteReq
      }
    }

    is(sWriteReq) {
      io.bankWrite(0).bank_id           := wbankReg
      io.bankWrite(0).group_id          := Mux(modeI8ToFp || modeI32Group, groupReg, 0.U)
      io.bankWrite(0).io.req.valid      := true.B
      io.bankWrite(0).io.req.bits.addr  := rowReg
      io.bankWrite(0).io.req.bits.data  := writeWord
      io.bankWrite(0).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
      io.bankWrite(0).io.req.bits.wmode := false.B
      when(io.bankWrite(0).io.req.fire) {
        state := sWriteResp
      }
    }

    is(sWriteResp) {
      io.bankWrite(0).bank_id       := wbankReg
      io.bankWrite(0).group_id      := Mux(modeI8ToFp || modeI32Group, groupReg, 0.U)
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(modeI8ToFp && groupReg =/= 3.U) {
          groupReg  := groupReg + 1.U
          writeWord := scaledI8Word(srcWord, groupReg + 1.U)
          state     := sWriteReq
        }.elsewhen(modeI32Group && groupReg =/= 3.U) {
          groupReg := groupReg + 1.U
          state    := sReadReq
        }.elsewhen(rowReg === iterReg - 1.U) {
          state := complete
        }.otherwise {
          rowReg   := rowReg + 1.U
          groupReg := 0.U
          state    := sReadReq
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
