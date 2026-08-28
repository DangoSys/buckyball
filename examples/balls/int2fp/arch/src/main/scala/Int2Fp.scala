package examples.balls.int2fp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.blink.mmio.MmioRead
import framework.top.GlobalConfig

/** INT32 accumulator dequantization: FP32 = INT32 * Da * Dw. */
@instantiable
class Int2Fp(val b: GlobalConfig) extends Module {
  val bankWidth    = b.memDomain.bankWidth
  require(bankWidth == 128, s"Int2FpBall requires bankWidth = 128, got $bankWidth")
  val elemsPerWord = bankWidth / 32

  val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Int2FpBall")
    .getOrElse(throw new IllegalArgumentException("Int2FpBall not found in config"))

  val inBW       = ballMapping.inBW
  val outBW      = ballMapping.outBW
  val mmioReadBW = ballMapping.mmioReadBW
  require(inBW >= 1 && outBW >= 1, "Int2Fp requires bank read/write lines")
  require(mmioReadBW == 4, "Int2Fp requires four MMIO byte read lines")

  val tensorFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "INT2FP_TENSOR")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("INT2FP_TENSOR not found in ballISA"))

  val channelFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "INT2FP_CHANNEL")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("INT2FP_CHANNEL not found in ballISA"))

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val mmioRead  = Vec(mmioReadBW, Flipped(new MmioRead(b)))
    val status    = new BallStatus
  })

  val idle :: sActReq :: sActResp :: sTensorReq :: sTensorResp :: sReadReq :: sReadResp :: sChannelReq :: sChannelResp :: sWriteReq :: sWriteResp :: complete :: Nil =
    Enum(12)
  val state                                                                                                                                                          = RegInit(idle)

  val robIdReg      = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSubReg      = RegInit(false.B)
  val subRobIdReg   = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val rbankReg      = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbankReg      = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iterReg       = RegInit(0.U(b.frontend.iter_len.W))
  val actAddrReg    = RegInit(0.U(log2Ceil(b.memDomain.mmioTotalBytes).W))
  val weightAddrReg = RegInit(0.U(log2Ceil(b.memDomain.mmioTotalBytes).W))
  val perChannelReg = RegInit(false.B)
  val daReg         = RegInit(0.U(32.W))
  val tensorDwReg   = RegInit(0.U(32.W))
  val rowReg        = RegInit(0.U(b.frontend.iter_len.W))
  val groupReg      = RegInit(0.U(2.W))
  val lastGroupReg  = RegInit(0.U(2.W))
  val laneReg       = RegInit(0.U(2.W))
  val srcWordReg    = RegInit(0.U(bankWidth.W))
  val writeWordReg  = RegInit(0.U(bankWidth.W))

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id           := robIdReg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := rbankReg
    io.bankRead(i).group_id         := groupReg
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id           := robIdReg
    io.bankWrite(i).ball_id          := 0.U
    io.bankWrite(i).bank_id          := wbankReg
    io.bankWrite(i).group_id         := groupReg
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready    := false.B
  }
  for (i <- 0 until mmioReadBW) {
    io.mmioRead(i).rob_id        := robIdReg
    io.mmioRead(i).ball_id       := 0.U
    io.mmioRead(i).req.valid     := false.B
    io.mmioRead(i).req.bits.addr := 0.U
    io.mmioRead(i).resp.ready    := false.B
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg

  def fp32Multiply(a: UInt, bv: UInt): UInt = {
    val aSign   = a(31); val bSign                   = bv(31)
    val aExp    = a(30, 23); val bExp                = bv(30, 23)
    val aMant   = Cat(1.U(1.W), a(22, 0)); val bMant = Cat(1.U(1.W), bv(22, 0))
    val prod    = (aMant * bMant)(47, 0)
    val sig     = Wire(UInt(24.W)); val guard        = Wire(Bool()); val round = Wire(Bool()); val sticky = Wire(Bool());
    val norm    = Wire(UInt(2.W))
    when(prod(47)) { sig := prod(47, 24); guard := prod(23); round := prod(22); sticky := prod(21, 0).orR; norm := 1.U }
      .otherwise { sig := prod(46, 23); guard := prod(22); round := prod(21); sticky := prod(20, 0).orR; norm := 0.U }
    val rounded = sig +& (guard && (round || sticky || sig(0))).asUInt
    val exp     = aExp +& bExp +& norm +& rounded(24) - 127.U
    val result  = Wire(UInt(32.W))
    when((aExp === 0.U && a(22, 0) === 0.U) || (bExp === 0.U && bv(22, 0) === 0.U))(result := 0.U)
      .elsewhen(exp(9))(result := 0.U)
      .elsewhen(exp(8))(result := Cat(aSign ^ bSign, 255.U(8.W), 0.U(23.W)))
      .otherwise(result := Cat(aSign ^ bSign, exp(7, 0), Mux(rounded(24), rounded(23, 1), rounded(22, 0))))
    result
  }

  def int32ToFp32(value: UInt): UInt = {
    val sign        = value(31)
    val abs         = Mux(sign, (~value).asUInt + 1.U, value)
    val zero        = abs === 0.U
    val leading     = 31.U - PriorityEncoder(Reverse(abs))
    val shift       = Mux(leading > 23.U, leading - 23.U, 0.U)
    val truncated   = abs >> shift
    val half        = Mux(shift === 0.U, 0.U(32.W), 1.U(32.W) << (shift - 1.U))
    val remainder   = Mux(shift === 0.U, 0.U(32.W), abs & ((1.U(32.W) << shift) - 1.U))
    val roundUp     = remainder > half || (remainder === half && truncated(0))
    val rounded     = truncated +& roundUp.asUInt
    val carry       = rounded(24)
    val significand = Mux(leading > 23.U, rounded(22, 0), (abs << (23.U - leading))(22, 0))
    val exponent    = (leading +& 127.U +& carry)(7, 0)
    Mux(zero, 0.U, Cat(sign, exponent, Mux(carry, 0.U(23.W), significand)))
  }

  def scaleResponse: UInt = Cat(io.mmioRead.reverse.map(_.resp.bits.data))
  def isFinitePositiveScale(bits: UInt): Bool =
    !bits(31) && bits(30, 23) =/= 255.U && bits(30, 0) =/= 0.U
  def allMmioReqFire:  Bool = io.mmioRead.map(_.req.fire).reduceLeft((x: Bool, y: Bool) => x && y)
  def allMmioRespFire: Bool = io.mmioRead.map(_.resp.fire).reduceLeft((x: Bool, y: Bool) => x && y)

  def driveScaleReq(addr: UInt): Unit = for (i <- 0 until mmioReadBW) {
    io.mmioRead(i).req.valid     := true.B
    io.mmioRead(i).req.bits.addr := addr + i.U
  }

  def driveScaleResp(): Unit = for (i <- 0 until mmioReadBW) io.mmioRead(i).resp.ready := true.B

  def dequantLane(word: UInt, lane: UInt, dw: UInt): UInt = {
    val acc = (word >> (lane << 5.U))(31, 0)
    fp32Multiply(int32ToFp32(acc), fp32Multiply(daReg, dw))
  }

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        assert(io.cmdReq.bits.cmd.iter > 0.U, "Int2Fp iter must be > 0")
        assert(
          io.cmdReq.bits.cmd.op1_col === io.cmdReq.bits.cmd.wr_col && io.cmdReq.bits.cmd.op1_col >= 1.U && io.cmdReq.bits.cmd.op1_col <= 4.U,
          "Int2Fp requires matching 1..4 accumulator/output groups"
        )
        assert(io.cmdReq.bits.cmd.op1_bank =/= io.cmdReq.bits.cmd.wr_bank, "Int2Fp forbids in-place dequantization")
        assert(
          io.cmdReq.bits.cmd.funct7 === tensorFunct.U(7.W) ||
            io.cmdReq.bits.cmd.funct7 === channelFunct.U(7.W),
          "Int2Fp funct7 must be INT2FP_TENSOR or INT2FP_CHANNEL"
        )
        assert(io.cmdReq.bits.cmd.special(63, 26) === 0.U, "Int2Fp reserves special[63:26]")
        assert(io.cmdReq.bits.cmd.special(12, 0) === 0.U, "Int2Fp Da address must be 0")
        assert(io.cmdReq.bits.cmd.special(25, 13) >= 16.U, "Int2Fp Dw address must be >= 16")
        assert(io.cmdReq.bits.cmd.special(1, 0) === 0.U, "Int2Fp Da address must be 4-byte aligned")
        assert(io.cmdReq.bits.cmd.special(14, 13) === 0.U, "Int2Fp Dw address must be 4-byte aligned")
        assert(
          (io.cmdReq.bits.cmd.special(12, 0) +& 3.U) < b.memDomain.mmioTotalBytes.U,
          "Int2Fp Da address exceeds MMIO space"
        )
        assert(
          (io.cmdReq.bits.cmd.special(25, 13) +& 3.U) < b.memDomain.mmioTotalBytes.U,
          "Int2Fp Dw address exceeds MMIO space"
        )
        when(io.cmdReq.bits.cmd.funct7 === channelFunct.U(7.W)) {
          assert(
            (io.cmdReq.bits.cmd.special(25, 13) +& (io.cmdReq.bits.cmd.op1_col << 4.U)) <= b.memDomain.mmioTotalBytes.U,
            "Int2Fp channel Dw range exceeds MMIO space"
          )
        }
        robIdReg      := io.cmdReq.bits.rob_id; isSubReg                           := io.cmdReq.bits.is_sub; subRobIdReg := io.cmdReq.bits.sub_rob_id
        rbankReg      := io.cmdReq.bits.cmd.op1_bank; wbankReg                     := io.cmdReq.bits.cmd.wr_bank;
        iterReg       := io.cmdReq.bits.cmd.iter
        actAddrReg    := io.cmdReq.bits.cmd.special(12, 0); weightAddrReg          := io.cmdReq.bits.cmd.special(25, 13)
        perChannelReg := io.cmdReq.bits.cmd.funct7 === channelFunct.U(7.W); rowReg := 0.U; groupReg                      := 0.U;
        laneReg       := 0.U
        lastGroupReg  := io.cmdReq.bits.cmd.op1_col - 1.U; writeWordReg            := 0.U
        state         := sActReq
      }
    }
    is(sActReq) { driveScaleReq(actAddrReg); when(allMmioReqFire)(state := sActResp) }
    is(sActResp) {
      driveScaleResp()
      when(allMmioRespFire) {
        assert(isFinitePositiveScale(scaleResponse), "Int2Fp Da must be finite and positive")
        daReg := scaleResponse
        state := sTensorReq
      }
    }
    is(sTensorReq) {
      when(perChannelReg)(state := sReadReq)
        .otherwise { driveScaleReq(weightAddrReg); when(allMmioReqFire)(state := sTensorResp) }
    }
    is(sTensorResp) {
      driveScaleResp()
      when(allMmioRespFire) {
        assert(isFinitePositiveScale(scaleResponse), "Int2Fp Dw must be finite and positive")
        tensorDwReg := scaleResponse
        state       := sReadReq
      }
    }
    is(sReadReq) {
      io.bankRead(0).io.req.valid            := true.B; io.bankRead(0).io.req.bits.addr := rowReg
      when(io.bankRead(0).io.req.fire)(state := sReadResp)
    }
    is(sReadResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        srcWordReg := io.bankRead(0).io.resp.bits.data; laneReg := 0.U; writeWordReg := 0.U
        when(perChannelReg)(state := sChannelReq)
          .otherwise {
            val result =
              Cat((0 until elemsPerWord).reverse.map(i => dequantLane(io.bankRead(0).io.resp.bits.data, i.U, tensorDwReg)))
            writeWordReg := result
            state        := sWriteReq
          }
      }
    }
    is(sChannelReq) {
      val channelAddr = weightAddrReg + ((groupReg * 4.U + laneReg) << 2.U)
      assert(channelAddr +& 3.U < b.memDomain.mmioTotalBytes.U, "Int2Fp channel Dw address exceeds MMIO space")
      driveScaleReq(channelAddr)
      when(allMmioReqFire)(state := sChannelResp)
    }
    is(sChannelResp) {
      driveScaleResp()
      when(allMmioRespFire) {
        assert(isFinitePositiveScale(scaleResponse), "Int2Fp Dw must be finite and positive")
        val next = writeWordReg | (dequantLane(srcWordReg, laneReg, scaleResponse) << (laneReg << 5.U))
        writeWordReg := next
        when(laneReg === 3.U)(state := sWriteReq).otherwise { laneReg := laneReg + 1.U; state := sChannelReq }
      }
    }
    is(sWriteReq) {
      io.bankWrite(0).io.req.valid            := true.B; io.bankWrite(0).io.req.bits.addr := rowReg
      io.bankWrite(0).io.req.bits.data        := writeWordReg;
      io.bankWrite(0).io.req.bits.mask        := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
      when(io.bankWrite(0).io.req.fire)(state := sWriteResp)
    }
    is(sWriteResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(groupReg === lastGroupReg && rowReg === iterReg - 1.U)(state := complete)
          .elsewhen(groupReg === lastGroupReg) { groupReg := 0.U; rowReg := rowReg + 1.U; state := sReadReq }
          .otherwise { groupReg := groupReg + 1.U; state := sReadReq }
      }
    }
    is(complete)(when(io.cmdResp.fire)(state := idle))
  }

  io.status.idle    := state === idle
  io.status.running := state =/= idle && state =/= complete
}
