package examples.balls.int2fp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import examples.balls.common.Fp32Mul
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class Int2Fp(val b: GlobalConfig) extends Module {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Int2FpBall")
    .getOrElse(throw new IllegalArgumentException("Int2FpBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "INT32_TO_FP32")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("INT32_TO_FP32 not found in ballISA"))

  require(b.memDomain.bankWidth == 128, "Int2FpBall requires 128-bit bank rows")
  require(mapping.inBW == 2, "Int2FpBall requires inBW=2")
  require(mapping.outBW == 1, "Int2FpBall requires outBW=1")
  require((funct >> 4) == 4, "INT32_TO_FP32 must encode two reads and one write")

  @public val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(mapping.inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(mapping.outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  private val idle :: readReq :: readResp :: convert :: multiply :: writeReq :: writeResp :: complete :: Nil =
    Enum(8)
  private val state                                                                                          = RegInit(idle)

  private val robIdReg    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSubReg    = RegInit(false.B)
  private val subRobIdReg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val inputBank   = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val scaleBank   = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iterReg     = RegInit(0.U(b.frontend.iter_len.W))
  private val rowReg      = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val reluReg     = RegInit(false.B)
  private val inputData   = Reg(UInt(128.W))
  private val scaleData   = Reg(UInt(128.W))
  private val fpOutputs   = Reg(Vec(4, UInt(32.W)))
  private val mulStarted  = RegInit(false.B)
  private val laneReg     = RegInit(0.U(2.W))
  private val writeData   = Reg(UInt(128.W))
  private val multiplier  = Module(new Fp32Mul)

  private def int32ToFp32(value: UInt): UInt = {
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

  private def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  for (i <- 0 until mapping.inBW) {
    io.bankRead(i).rob_id           := robIdReg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := Mux(i.U === 0.U, inputBank, scaleBank)
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := Mux(i.U === 0.U, rowReg, rowReg(1, 0))
    io.bankRead(i).io.resp.ready    := false.B
  }
  io.bankWrite(0).rob_id := robIdReg
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := 0.U
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := rowReg
  io.bankWrite(0).io.req.bits.data := writeData
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  private val inputValue = (inputData >> (laneReg << 5))(31, 0)
  private val scaleValue = (scaleData >> (laneReg << 5))(31, 0)
  multiplier.io.start := false.B
  multiplier.io.a     := int32ToFp32(Mux(reluReg && inputValue(31), 0.U, inputValue))
  multiplier.io.b     := scaleValue

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        assert(cmd.funct7 === funct.U, "Int2FpBall received an unknown funct7")
        assert(cmd.iter > 0.U && cmd.iter(1, 0) === 0.U, "INT32_TO_FP32 iter must be a positive multiple of four")
        assert(cmd.iter <= b.memDomain.bankEntries.U, "INT32_TO_FP32 input exceeds bank depth")
        assert(
          cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
          "INT32_TO_FP32 operands must each occupy one bank"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "INT32_TO_FP32 banks must be distinct"
        )
        assert(cmd.rs2(63, 1) === 0.U, "INT32_TO_FP32 reserves rs2[63:1]")

        robIdReg    := io.cmdReq.bits.rob_id
        isSubReg    := io.cmdReq.bits.is_sub
        subRobIdReg := io.cmdReq.bits.sub_rob_id
        inputBank   := cmd.op1_bank
        scaleBank   := cmd.op2_bank
        outputBank  := cmd.wr_bank
        iterReg     := cmd.iter
        rowReg      := 0.U
        reluReg     := cmd.rs2(0)
        state       := readReq
      }
    }
    is(readReq) {
      for (i <- 0 until mapping.inBW) io.bankRead(i).io.req.valid := true.B
      when(io.bankRead.map(_.io.req.fire).reduce(_ && _)) {
        state := readResp
      }
    }
    is(readResp) {
      for (i <- 0 until mapping.inBW) io.bankRead(i).io.resp.ready := true.B
      when(io.bankRead.map(_.io.resp.fire).reduce(_ && _)) {
        inputData := io.bankRead(0).io.resp.bits.data
        scaleData := io.bankRead(1).io.resp.bits.data
        state     := convert
      }
    }
    is(convert) {
      for (i <- 0 until 4) {
        val scale = scaleData(32 * i + 31, 32 * i)
        assert(positiveFinite(scale), "INT32_TO_FP32 scales must be finite and positive")
      }
      mulStarted := false.B
      laneReg := 0.U
      state   := multiply
    }
    is(multiply) {
      when(!mulStarted) {
        multiplier.io.start := true.B
        mulStarted          := true.B
      }
      when(multiplier.io.done) {
        fpOutputs(laneReg) := multiplier.io.result
        when(laneReg === 3.U) {
          writeData := Cat(multiplier.io.result, fpOutputs(2), fpOutputs(1), fpOutputs(0))
          state     := writeReq
        }.otherwise {
          laneReg    := laneReg + 1.U
          mulStarted := false.B
        }
      }
    }
    is(writeReq) {
      io.bankWrite(0).io.req.valid := true.B
      when(io.bankWrite(0).io.req.fire) {
        state := writeResp
      }
    }
    is(writeResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(rowReg === iterReg - 1.U) {
          state := complete
        }.otherwise {
          rowReg := rowReg + 1.U
          state  := readReq
        }
      }
    }
    is(complete) {
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }
}
