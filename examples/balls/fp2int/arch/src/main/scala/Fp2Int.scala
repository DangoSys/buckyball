package examples.balls.fp2int

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.blink.mmio.MmioWrite
import framework.top.GlobalConfig

@instantiable
class Fp2Int(val b: GlobalConfig) extends Module {
  val bankWidth    = b.memDomain.bankWidth
  require(
    bankWidth == 128,
    s"Fp2IntBall requires bankWidth = 128, got $bankWidth"
  )
  val elemsPerWord = bankWidth / 32

  val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Fp2IntBall")
    .getOrElse(
      throw new IllegalArgumentException("Fp2IntBall not found in config")
    )

  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW
  val mmioWriteBW = ballMapping.mmioWriteBW

  require(inBW >= 1, "Fp2Int requires at least one read port")
  require(outBW >= 1, "Fp2Int requires at least one write port")
  require(mmioWriteBW == 4, "Fp2Int requires four MMIO byte write lines")

  val fp2IntFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "FP2INT")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("FP2INT not found in ballISA"))

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val mmioWrite = Vec(mmioWriteBW, Flipped(new MmioWrite(b)))
    val status    = new BallStatus
  })

  val idle :: sScanReq :: sScanResp :: sScaleWrite :: sReadReq :: sReadResp :: sWriteReq :: sWriteResp :: complete :: Nil =
    Enum(9)
  val state                                                                                                               = RegInit(idle)

  val robIdReg    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSubReg    = RegInit(false.B)
  val subRobIdReg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  val rbankReg      = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbankReg      = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iterReg       = RegInit(0.U(b.frontend.iter_len.W))
  val scaleReg      = RegInit(0.U(32.W))
  val quantScaleReg = RegInit(0.U(32.W))
  val scaleAddrReg  = RegInit(0.U(log2Ceil(b.memDomain.mmioTotalBytes).W))
  val scanMaxReg    = RegInit(0.U(31.W))

  val rowReg      = RegInit(0.U(b.frontend.iter_len.W))
  val groupReg    = RegInit(0.U(5.W))
  val srcColReg   = RegInit(0.U(5.W))
  val dstColReg   = RegInit(0.U(5.W))
  val packWordReg = RegInit(0.U(2.W))
  val outRowReg   = RegInit(0.U(b.frontend.iter_len.W))
  val outGroupReg = RegInit(0.U(5.W))
  val outWord     = RegInit(0.U(bankWidth.W))
  val writeWord   = RegInit(0.U(bankWidth.W))

  for (i <- 0 until mmioWriteBW) {
    io.mmioWrite(i).rob_id        := robIdReg
    io.mmioWrite(i).ball_id       := 0.U
    io.mmioWrite(i).req.valid     := false.B
    io.mmioWrite(i).req.bits.addr := 0.U
    io.mmioWrite(i).req.bits.data := 0.U
  }

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
    io.bankWrite(i).io.req.bits.mask := VecInit(
      Seq.fill(b.memDomain.bankMaskLen)(false.B)
    )
    io.bankWrite(i).io.resp.ready    := false.B
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg

  def fp32Multiply(a: UInt, bv: UInt): UInt = {
    val aSign      = a(31)
    val bSign      = bv(31)
    val aExp       = a(30, 23)
    val bExp       = bv(30, 23)
    val aMant      = Cat(1.U(1.W), a(22, 0))
    val bMant      = Cat(1.U(1.W), bv(22, 0))
    val resSign    = aSign ^ bSign
    val aZero      = aExp === 0.U && a(22, 0) === 0.U
    val bZero      = bExp === 0.U && bv(22, 0) === 0.U
    val prod       = (aMant * bMant)(47, 0)
    val sig        = Wire(UInt(24.W))
    val guard      = Wire(Bool())
    val round      = Wire(Bool())
    val sticky     = Wire(Bool())
    val normAdjust = Wire(UInt(2.W))
    when(prod(47)) {
      sig        := prod(47, 24)
      guard      := prod(23)
      round      := prod(22)
      sticky     := prod(21, 0).orR
      normAdjust := 1.U
    }.otherwise {
      sig        := prod(46, 23)
      guard      := prod(22)
      round      := prod(21)
      sticky     := prod(20, 0).orR
      normAdjust := 0.U
    }
    val increment  = guard && (round || sticky || sig(0))
    val rounded    = sig +& increment.asUInt
    val finalSig   = Mux(rounded(24), rounded(24, 1), rounded(23, 0))
    val expWide    = aExp +& bExp +& normAdjust +& rounded(24) - 127.U
    val result     = Wire(UInt(32.W))
    when(aZero || bZero) {
      result := 0.U
    }.elsewhen(expWide(9, 8) =/= 0.U && expWide(9)) {
      result := 0.U
    }.elsewhen(expWide(8) && !expWide(9)) {
      result := Cat(resSign, 255.U(8.W), 0.U(23.W))
    }.otherwise {
      result := Cat(resSign, expWide(7, 0), finalSig(22, 0))
    }
    result
  }

  def fp32Divide(a: UInt, bv: UInt): UInt = {
    val aExp          = a(30, 23)
    val bExp          = bv(30, 23)
    val aMant         = Cat(1.U(1.W), a(22, 0))
    val bMant         = Cat(1.U(1.W), bv(22, 0))
    val normalizeDown = aMant < bMant
    val dividend      = Mux(normalizeDown, aMant << 26.U, aMant << 25.U)
    val quotient      = dividend / bMant
    val remainder     = dividend % bMant
    val sig           = quotient >> 2.U
    val roundUp       = quotient(1) && (quotient(0) || remainder.orR || sig(0))
    val rounded       = sig +& roundUp.asUInt
    val exp           = aExp.zext - bExp.zext + Mux(normalizeDown, 126.S, 127.S) + rounded(24).asSInt
    val result        = Wire(UInt(32.W))
    when(a(30, 0) === 0.U) {
      result := 0.U
    }.elsewhen(bv(30, 0) === 0.U) {
      result := Cat(a(31) ^ bv(31), 255.U(8.W), 0.U(23.W))
    }.elsewhen(exp < 1.S) {
      result := 0.U
    }.elsewhen(exp > 254.S) {
      result := Cat(a(31) ^ bv(31), 255.U(8.W), 0.U(23.W))
    }.otherwise {
      result := Cat(a(31) ^ bv(31), exp.asUInt(7, 0), Mux(rounded(24), 0.U(23.W), rounded(22, 0)))
    }
    result
  }

  def fp32ToInt32(fp: UInt): UInt = {
    val sign         = fp(31)
    val exponent     = fp(30, 23)
    val fraction     = fp(22, 0)
    val mantissa     =
      Mux(exponent === 0.U, Cat(0.U(1.W), fraction), Cat(1.U(1.W), fraction))
    val mantissaWide = Cat(0.U(40.W), mantissa)
    val expVal       = exponent.zext - 127.S
    val magnitude    = Wire(UInt(64.W))

    magnitude := 0.U
    when(expVal >= 31.S) {
      magnitude := "h80000000".U
    }.elsewhen(expVal >= 23.S) {
      magnitude := mantissaWide << (expVal - 23.S).asUInt
    }.elsewhen(expVal >= -1.S) {
      val rightShift = (23.S - expVal).asUInt
      val truncated  = mantissaWide >> rightShift
      val half       = 1.U(64.W) << (rightShift - 1.U)
      val remainder  = mantissaWide & ((1.U(64.W) << rightShift) - 1.U)
      val roundUp    = remainder > half || (remainder === half && truncated(0))
      magnitude := truncated + roundUp.asUInt
    }

    val result = Wire(SInt(32.W))
    when(exponent === 255.U && fraction =/= 0.U) {
      result := 0.S
    }.elsewhen(sign.asBool) {
      result := Mux(
        magnitude >= "h80000000".U,
        -2147483648L.S(32.W),
        -magnitude(31, 0).asSInt
      )
    }.otherwise {
      result := Mux(
        magnitude > "h7fffffff".U,
        2147483647.S(32.W),
        magnitude(31, 0).asSInt
      )
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

  def quantWord(data: UInt): UInt = {
    val i8 = Wire(Vec(elemsPerWord, UInt(8.W)))
    for (i <- 0 until elemsPerWord) {
      val fp     = data((i + 1) * 32 - 1, i * 32)
      val scaled = fp32Multiply(fp, quantScaleReg)
      i8(i) := fp32ToInt8(scaled)
    }
    Cat(i8.reverse)
  }

  def wordMaxAbs(data: UInt): UInt = {
    val values = Seq.tabulate(elemsPerWord) { i =>
      val fp = data((i + 1) * 32 - 1, i * 32)
      assert(fp(30, 23) =/= 255.U, "Fp2Int does not accept NaN or infinity")
      fp(30, 0)
    }
    values.reduceLeft((a, b) => Mux(a > b, a, b))
  }

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        assert(io.cmdReq.bits.cmd.funct7 === fp2IntFunct.U(7.W), "Fp2Int funct7 must be FP2INT")
        val srcCol      = io.cmdReq.bits.cmd.op1_col
        val dstCol      = io.cmdReq.bits.cmd.wr_col
        val sourceWords = io.cmdReq.bits.cmd.iter * srcCol
        val dstRowWords = Cat(dstCol, 0.U(2.W))

        assert(io.cmdReq.bits.cmd.iter > 0.U, "Fp2Int iter must be > 0")
        assert(srcCol > 0.U && dstCol > 0.U, "Fp2Int source and destination groups must be nonzero")
        assert(sourceWords % dstRowWords === 0.U, "Fp2Int source stream must fill destination rows")
        assert(sourceWords / dstRowWords <= b.memDomain.bankEntries.U, "Fp2Int destination exceeds bank capacity")
        assert(io.cmdReq.bits.cmd.op1_bank =/= io.cmdReq.bits.cmd.wr_bank, "Fp2Int forbids in-place quantization")
        assert(io.cmdReq.bits.cmd.special(63, 13) === 0.U, "Fp2Int reserves special[63:13]")
        assert(io.cmdReq.bits.cmd.special(12, 0) === 0.U, "Fp2Int Da address must be 0")
        assert(io.cmdReq.bits.cmd.special(1, 0) === 0.U, "Fp2Int Da address must be 4-byte aligned")
        assert(
          (io.cmdReq.bits.cmd.special(12, 0) +& 3.U) < b.memDomain.mmioTotalBytes.U,
          "Fp2Int Da address exceeds MMIO space"
        )

        robIdReg     := io.cmdReq.bits.rob_id
        isSubReg     := io.cmdReq.bits.is_sub
        subRobIdReg  := io.cmdReq.bits.sub_rob_id
        rbankReg     := io.cmdReq.bits.cmd.op1_bank
        wbankReg     := io.cmdReq.bits.cmd.wr_bank
        iterReg      := io.cmdReq.bits.cmd.iter
        srcColReg    := srcCol
        dstColReg    := dstCol
        scaleAddrReg := io.cmdReq.bits.cmd.special(12, 0)
        scanMaxReg   := 0.U
        rowReg       := 0.U
        groupReg     := 0.U
        packWordReg  := 0.U
        outRowReg    := 0.U
        outGroupReg  := 0.U
        outWord      := 0.U
        writeWord    := 0.U
        state        := sScanReq
      }
    }

    is(sScanReq) {
      io.bankRead(0).bank_id          := rbankReg
      io.bankRead(0).group_id         := groupReg
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := rowReg
      when(io.bankRead(0).io.req.fire) {
        state := sScanResp
      }
    }

    is(sScanResp) {
      io.bankRead(0).bank_id       := rbankReg
      io.bankRead(0).group_id      := groupReg
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        val nextMax = Mux(
          wordMaxAbs(io.bankRead(0).io.resp.bits.data) > scanMaxReg,
          wordMaxAbs(io.bankRead(0).io.resp.bits.data),
          scanMaxReg
        )
        scanMaxReg := nextMax
        when(groupReg === srcColReg - 1.U && rowReg === iterReg - 1.U) {
          val da = Mux(nextMax === 0.U, "h3f800000".U, fp32Divide(Cat(0.U(1.W), nextMax), "h42fe0000".U))
          scaleReg      := da
          quantScaleReg := fp32Divide("h3f800000".U, da)
          rowReg        := 0.U
          groupReg      := 0.U
          state         := sScaleWrite
        }.elsewhen(groupReg === srcColReg - 1.U) {
          groupReg := 0.U
          rowReg   := rowReg + 1.U
          state    := sScanReq
        }.otherwise {
          groupReg := groupReg + 1.U
          state    := sScanReq
        }
      }
    }

    is(sScaleWrite) {
      for (i <- 0 until mmioWriteBW) {
        io.mmioWrite(i).req.valid     := true.B
        io.mmioWrite(i).req.bits.addr := scaleAddrReg + i.U
        io.mmioWrite(i).req.bits.data := scaleReg((i + 1) * 8 - 1, i * 8)
      }
      when(io.mmioWrite.map(_.req.fire).reduceLeft((a: Bool, c: Bool) => a && c)) {
        state := sReadReq
      }
    }

    is(sReadReq) {
      io.bankRead(0).bank_id          := rbankReg
      io.bankRead(0).group_id         := groupReg
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := rowReg
      when(io.bankRead(0).io.req.fire) {
        state := sReadResp
      }
    }

    is(sReadResp) {
      io.bankRead(0).bank_id       := rbankReg
      io.bankRead(0).group_id      := groupReg
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        val i8Bytes  = quantWord(io.bankRead(0).io.resp.bits.data)
        val nextWord = outWord | (i8Bytes << (packWordReg << 5.U))
        when(packWordReg === 3.U) {
          writeWord   := nextWord
          packWordReg := 0.U
          outWord     := 0.U
          state       := sWriteReq
        }.otherwise {
          outWord     := nextWord
          packWordReg := packWordReg + 1.U
          when(groupReg === srcColReg - 1.U) {
            groupReg := 0.U
            rowReg   := rowReg + 1.U
          }.otherwise {
            groupReg := groupReg + 1.U
          }
          state       := sReadReq
        }
      }
    }

    is(sWriteReq) {
      io.bankWrite(0).bank_id          := wbankReg
      io.bankWrite(0).group_id         := outGroupReg
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := outRowReg
      io.bankWrite(0).io.req.bits.data := writeWord
      io.bankWrite(0).io.req.bits.mask := VecInit(
        Seq.fill(b.memDomain.bankMaskLen)(true.B)
      )
      when(io.bankWrite(0).io.req.fire) {
        state := sWriteResp
      }
    }

    is(sWriteResp) {
      io.bankWrite(0).bank_id       := wbankReg
      io.bankWrite(0).group_id      := outGroupReg
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(rowReg === iterReg - 1.U && groupReg === srcColReg - 1.U) {
          state := complete
        }.otherwise {
          when(groupReg === srcColReg - 1.U) {
            groupReg := 0.U
            rowReg   := rowReg + 1.U
          }.otherwise {
            groupReg := groupReg + 1.U
          }
          when(outGroupReg === dstColReg - 1.U) {
            outGroupReg := 0.U
            outRowReg   := outRowReg + 1.U
          }.otherwise {
            outGroupReg := outGroupReg + 1.U
          }
          state := sReadReq
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
