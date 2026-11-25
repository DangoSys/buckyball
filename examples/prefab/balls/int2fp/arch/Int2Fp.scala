package examples.balls.int2fp

import framework.generic.contract.blink.BlinkParam

import framework.generic.contract.blink.BallResponse

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import examples.balls.int2fp.configs.Int2FpBallParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.ball.blink.BallStatus
import framework.generic.arith.Fma
import hardfloat.{fNFromRecFN, recFNFromFN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}
import framework.generic.contract.configs.BallRouteParam
import framework.generic.contract.blink.BallCommand
import framework.generic.contract.memdomain.BankRead
import framework.generic.contract.memdomain.BankWrite

@instantiable
class Int2Fp(
  link:          BlinkParam,
  instance:      BallRouteParam,
  val ballParam: Int2FpBallParam)
    extends Module {
  val commandDataBits = link.dataBits

  val mapping = instance

  require(link.bankWidth == 128, "Int2FpBall requires 128-bit bank rows")
  require(mapping.inBW == 2, "Int2FpBall requires inBW=2")
  require(mapping.outBW == 1, "Int2FpBall requires outBW=1")

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(link)))

    val cmdResp = Decoupled(new BallResponse(link))

    val bankRead = Vec(
      mapping.inBW,
      Flipped(new BankRead(
        link
      ))
    )

    val bankWrite = Vec(
      mapping.outBW,
      Flipped(new BankWrite(
        link
      ))
    )

    val status = new BallStatus
  })

  val idle :: readReq :: readResp :: convert :: multiply :: writeReq :: writeResp :: complete :: Nil =
    Enum(8)
  val state                                                                                          = RegInit(idle)

  val robIdReg     = RegInit(0.U(link.robIdBits.W))
  val inputBank    = RegInit(0.U(link.bankIdBits.W))
  val scaleBank    = RegInit(0.U(link.bankIdBits.W))
  val outputBank   = RegInit(0.U(link.bankIdBits.W))
  val bases        = Reg(Vec(3, UInt(link.groupIdBits.W)))
  // Every group of a virtual bank is one 16-channel block with the same rows.
  val groups       = RegInit(0.U(log2Up(link.bankNum + 1).W))
  val group        = RegInit(0.U(log2Up(link.bankNum + 1).W))
  val iterReg      = RegInit(0.U(link.iterBits.W))
  val rowReg       = RegInit(0.U(log2Ceil(link.bankEntries).W))
  val reluReg      = RegInit(false.B)
  val inputData    = Reg(UInt(128.W))
  val scaleData    = Reg(UInt(128.W))
  val readSent     = RegInit(VecInit(Seq.fill(mapping.inBW)(false.B)))
  val readReceived = RegInit(VecInit(Seq.fill(mapping.inBW)(false.B)))
  val mulStarted   = RegInit(false.B)
  val writeData    = Reg(UInt(128.W))
  val converted    = Reg(Vec(4, UInt(32.W)))
  val multipliers  = Seq.fill(4)(Instantiate(new Fma(3, 8, 24)))

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

  def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  for (i <- 0 until mapping.inBW) {
    io.bankRead(i).rob_id           := robIdReg
    io.bankRead(i).bank_id          := Mux(i.U === 0.U, inputBank, scaleBank)
    io.bankRead(i).group_id         := bases(i) + group
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := Mux(i.U === 0.U, rowReg, rowReg(1, 0))
    io.bankRead(i).io.resp.ready    := false.B
  }
  io.bankWrite(0).rob_id := robIdReg
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := bases(2) + group
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := rowReg
  io.bankWrite(0).io.req.bits.data := writeData
  io.bankWrite(0).io.req.bits.mask := VecInit(
    Seq.fill(link.bankMaskLen)(true.B)
  )
  io.bankWrite(0).io.resp.ready    := false.B

  for (lane <- 0 until 4) {
    multipliers(lane).io.a              := recFNFromFN(8, 24, converted(lane))
    multipliers(lane).io.b              := recFNFromFN(8, 24, scaleData(32 * lane + 31, 32 * lane))
    multipliers(lane).io.c              := recFNFromFN(8, 24, 0.U(32.W))
    multipliers(lane).io.op             := 0.U
    multipliers(lane).io.validin        := state === multiply && !mulStarted
    multipliers(lane).io.roundingMode   := round_near_even
    multipliers(lane).io.detectTininess := tininess_afterRounding.asUInt
  }

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := state === complete
  io.cmdResp.bits.rob_id := robIdReg
  io.status.idle         := state === idle
  io.status.running      := state =/= idle && state =/= complete

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits

        assert(cmd.iter > 0.U && cmd.iter(1, 0) === 0.U, "INT32_TO_FP32 iter must be a positive multiple of four")
        assert(
          cmd.iter <= link.bankEntries.U,
          "INT32_TO_FP32 input exceeds bank depth"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op1_col === cmd.op2_col && cmd.op1_col === cmd.wr_col &&
            cmd.op1_col <= link.bankNum.U,
          "INT32_TO_FP32 operands must have the same channel groups"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "INT32_TO_FP32 banks must be distinct"
        )
        assert(cmd.rs2(63, 1) === 0.U, "INT32_TO_FP32 reserves rs2[63:1]")

        robIdReg               := io.cmdReq.bits.rob_id
        inputBank              := cmd.op1_bank
        scaleBank              := cmd.op2_bank
        outputBank             := cmd.wr_bank
        bases                  := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        groups                 := cmd.op1_col
        group                  := 0.U
        iterReg                := cmd.iter
        rowReg                 := 0.U
        reluReg                := cmd.rs2(0)
        readSent.foreach(_     := false.B)
        readReceived.foreach(_ := false.B)
        state                  := readReq
      }
    }
    is(readReq) {
      for (i <- 0 until mapping.inBW) {
        io.bankRead(i).io.req.valid                  := !readSent(i)
        when(io.bankRead(i).io.req.fire)(readSent(i) := true.B)
      }
      when(VecInit(io.bankRead.zipWithIndex.map { case (port, i) => readSent(i) || port.io.req.fire }).asUInt.andR) {
        state := readResp
      }
    }
    is(readResp) {
      for (i <- 0 until mapping.inBW) {
        io.bankRead(i).io.resp.ready                      := !readReceived(i)
        when(io.bankRead(i).io.resp.fire)(readReceived(i) := true.B)
      }
      when(io.bankRead(0).io.resp.fire)(inputData := io.bankRead(0).io.resp.bits.data)
      when(io.bankRead(1).io.resp.fire)(scaleData := io.bankRead(1).io.resp.bits.data)
      when(VecInit(io.bankRead.zipWithIndex.map { case (port, i) => readReceived(i) || port.io.resp.fire }).asUInt.andR) {
        state := convert
      }
    }
    is(convert) {
      for (i <- 0 until 4) {
        val scale = scaleData(32 * i + 31, 32 * i)
        assert(positiveFinite(scale), "INT32_TO_FP32 scales must be finite and positive")
        val input = inputData(32 * i + 31, 32 * i)
        converted(i) := int32ToFp32(Mux(reluReg && input(31), 0.U(32.W), input))
      }
      mulStarted := false.B
      state := multiply
    }
    is(multiply) {
      mulStarted := true.B
      when(multipliers.head.io.validout) {
        writeData := Cat(multipliers.reverse.map(m => fNFromRecFN(8, 24, m.io.out)))
        state     := writeReq
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
        when(rowReg === iterReg - 1.U && group + 1.U === groups) {
          state := complete
        }.otherwise {
          when(rowReg === iterReg - 1.U) {
            group  := group + 1.U
            rowReg := 0.U
          }.otherwise {
            rowReg := rowReg + 1.U
          }
          readSent.foreach(_     := false.B)
          readReceived.foreach(_ := false.B)
          state                  := readReq
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
