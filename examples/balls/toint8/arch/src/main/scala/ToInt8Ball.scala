package examples.balls.toint8

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import hardfloat.{recFNFromFN, INToRecFN, MulRecFN, RecFNToIN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}

import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class ToInt8Ball(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "ToInt8Ball")
    .getOrElse(throw new IllegalArgumentException("ToInt8Ball not found in config"))

  private val f32Funct = b.ballDomain.ballISA
    .find(_.mnemonic == "QUANT_F32_TO_I8")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("QUANT_F32_TO_I8 not found in ballISA"))

  private val i32Funct = b.ballDomain.ballISA
    .find(_.mnemonic == "QUANT_I32_TO_I8")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("QUANT_I32_TO_I8 not found in ballISA"))

  require(b.memDomain.bankWidth == 128, "ToInt8Ball requires 128-bit bank rows")
  require(b.memDomain.bankEntries >= 4, "ToInt8Ball requires at least four bank rows")
  require(mapping.inBW == 1, "ToInt8Ball requires inBW=1")
  require(mapping.outBW == 1, "ToInt8Ball requires outBW=1")
  require((f32Funct >> 4) == 3, "QUANT_F32_TO_I8 must encode one read and one write")
  require((i32Funct >> 4) == 4, "QUANT_I32_TO_I8 must encode two reads and one write")

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val idle :: scaleReq :: scaleResp :: inputReq :: inputResp :: pack :: writeReq :: writeResp :: complete :: Nil =
    Enum(9)
  private val state                                                                                                      = RegInit(idle)

  private val robIdReg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSubReg     = RegInit(false.B)
  private val subRobIdReg  = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val inputBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val scaleBank    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank   = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iterReg      = RegInit(0.U(b.frontend.iter_len.W))
  private val inputBaseReg = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val inputRow     = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val outputRow    = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val outputWidth  = RegInit(0.U(7.W))
  private val outputStride = RegInit(0.U(7.W))
  private val outputColumn = RegInit(0.U(7.W))
  private val scaleRow     = RegInit(0.U(2.W))
  private val i32Mode      = RegInit(false.B)
  private val reluReg      = RegInit(false.B)
  private val tensorScale  = RegInit(0.U(32.W))
  private val scales       = Reg(Vec(16, UInt(32.W)))
  private val inputData    = Reg(UInt(128.W))
  private val outputData   = RegInit(0.U(128.W))
  private val writeData    = Reg(UInt(128.W))

  private def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  private val intToFloat = Seq.fill(4)(Module(new INToRecFN(32, 8, 24)))
  private val multiply   = Seq.fill(4)(Module(new MulRecFN(8, 24)))
  private val floatToInt = Seq.fill(4)(Module(new RecFNToIN(8, 24, 8)))
  private val quantized  = Wire(Vec(4, UInt(8.W)))

  for (lane <- 0 until 4) {
    val input = inputData(32 * lane + 31, 32 * lane)
    val scale = Mux(i32Mode, scales(Cat(inputRow(1, 0), lane.U(2.W))), tensorScale)

    intToFloat(lane).io.signedIn       := true.B
    intToFloat(lane).io.in             := input
    intToFloat(lane).io.roundingMode   := round_near_even
    intToFloat(lane).io.detectTininess := tininess_afterRounding

    multiply(lane).io.a              := Mux(
      i32Mode,
      intToFloat(lane).io.out,
      recFNFromFN(8, 24, input)
    )
    multiply(lane).io.b              := recFNFromFN(8, 24, scale)
    multiply(lane).io.roundingMode   := round_near_even
    multiply(lane).io.detectTininess := tininess_afterRounding.asBool

    floatToInt(lane).io.in           := multiply(lane).io.out
    floatToInt(lane).io.roundingMode := round_near_even
    floatToInt(lane).io.signedOut    := true.B
    quantized(lane)                  := Mux(reluReg && floatToInt(lane).io.out(7), 0.U, floatToInt(lane).io.out)
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  io.bankRead(0).rob_id           := robIdReg
  io.bankRead(0).ball_id          := 0.U
  io.bankRead(0).bank_id          := Mux(state === scaleReq || state === scaleResp, scaleBank, inputBank)
  io.bankRead(0).group_id         := 0.U
  io.bankRead(0).io.req.valid     := false.B
  io.bankRead(0).io.req.bits.addr := 0.U
  io.bankRead(0).io.resp.ready    := false.B

  io.bankWrite(0).rob_id           := robIdReg
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := 0.U
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := outputRow
  io.bankWrite(0).io.req.bits.data := writeData
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd   = io.cmdReq.bits.cmd
        val isF32 = cmd.funct7 === f32Funct.U
        val isI32 = cmd.funct7 === i32Funct.U

        assert(isF32 || isI32, "ToInt8Ball received an unknown funct7")
        assert(cmd.iter > 0.U && cmd.iter(1, 0) === 0.U, "ToInt8Ball iter must be a positive multiple of four")
        assert(cmd.iter <= b.memDomain.bankEntries.U, "ToInt8Ball input exceeds bank depth")
        assert(cmd.rs1(9, 0) < b.memDomain.bankNum.U, "ToInt8Ball input bank is invalid")
        assert(cmd.rs1(29, 20) < b.memDomain.bankNum.U, "ToInt8Ball output bank is invalid")
        assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "ToInt8Ball input and output must each occupy one bank")
        assert(cmd.op1_bank =/= cmd.wr_bank, "ToInt8Ball input and output banks must differ")
        when(isF32) {
          assert(cmd.rs1(19, 10) === 0.U, "QUANT_F32_TO_I8 reserves input bank 1")
          assert(cmd.rs2(63, 32) === 0.U, "QUANT_F32_TO_I8 reserves rs2[63:32]")
          assert(positiveFinite(cmd.rs2(31, 0)), "QUANT_F32_TO_I8 scale must be finite and positive")
        }.otherwise {
          assert(cmd.rs1(19, 10) < b.memDomain.bankNum.U, "QUANT_I32_TO_I8 scale bank is invalid")
          assert(cmd.op2_col === 1.U, "QUANT_I32_TO_I8 scale must occupy one bank")
          assert(cmd.op1_bank =/= cmd.op2_bank, "QUANT_I32_TO_I8 input and scale banks must differ")
          assert(cmd.op2_bank =/= cmd.wr_bank, "QUANT_I32_TO_I8 scale and output banks must differ")
          val outputBase = cmd.rs2(7, 1)
          val width      = cmd.rs2(14, 8)
          val height     = cmd.rs2(21, 15)
          val stride     = cmd.rs2(28, 22)
          val outputRows = cmd.iter >> 2
          val outputEnd  = outputBase +& ((height - 1.U) * stride) +& width
          val inputBase  = cmd.rs2(34, 29)
          assert(cmd.rs2(63, 35) === 0.U, "QUANT_I32_TO_I8 reserves rs2[63:35]")
          assert(width > 0.U && height > 0.U && stride >= width, "QUANT_I32_TO_I8 output geometry is invalid")
          assert(width * height === outputRows, "QUANT_I32_TO_I8 iter does not match output geometry")
          assert(outputEnd <= b.memDomain.bankEntries.U, "QUANT_I32_TO_I8 output exceeds bank depth")
          assert(inputBase +& cmd.iter <= b.memDomain.bankEntries.U, "QUANT_I32_TO_I8 input exceeds bank depth")
        }

        robIdReg     := io.cmdReq.bits.rob_id
        isSubReg     := io.cmdReq.bits.is_sub
        subRobIdReg  := io.cmdReq.bits.sub_rob_id
        inputBank    := cmd.op1_bank
        scaleBank    := cmd.op2_bank
        outputBank   := cmd.wr_bank
        iterReg      := cmd.iter
        inputBaseReg := Mux(isI32, cmd.rs2(34, 29), 0.U)
        inputRow     := 0.U
        outputRow    := Mux(isI32, cmd.rs2(7, 1), 0.U)
        outputWidth  := Mux(isI32, cmd.rs2(14, 8), 0.U)
        outputStride := Mux(isI32, cmd.rs2(28, 22), 0.U)
        outputColumn := 0.U
        scaleRow     := 0.U
        i32Mode      := isI32
        reluReg      := isI32 && cmd.rs2(0)
        tensorScale  := cmd.rs2(31, 0)
        outputData   := 0.U
        state        := Mux(isI32, scaleReq, inputReq)
      }
    }

    is(scaleReq) {
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := scaleRow
      when(io.bankRead(0).io.req.fire) {
        state := scaleResp
      }
    }

    is(scaleResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        for (lane <- 0 until 4) {
          val value = io.bankRead(0).io.resp.bits.data(32 * lane + 31, 32 * lane)
          assert(positiveFinite(value), "QUANT_I32_TO_I8 scales must be finite and positive")
          scales(Cat(scaleRow, lane.U(2.W))) := value
        }
        when(scaleRow === 3.U) {
          state := inputReq
        }.otherwise {
          scaleRow := scaleRow + 1.U
          state    := scaleReq
        }
      }
    }

    is(inputReq) {
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := inputBaseReg +& inputRow
      when(io.bankRead(0).io.req.fire) {
        state := inputResp
      }
    }

    is(inputResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        inputData := io.bankRead(0).io.resp.bits.data
        state     := pack
      }
    }

    is(pack) {
      when(!i32Mode) {
        for (lane <- 0 until 4) {
          assert(inputData(32 * lane + 30, 32 * lane + 23) =/= 255.U, "QUANT_F32_TO_I8 input must be finite")
        }
      }
      val bytes    = Cat(quantized.reverse)
      val nextWord = outputData | (bytes << (inputRow(1, 0) << 5))
      when(inputRow(1, 0) === 3.U) {
        writeData  := nextWord
        outputData := 0.U
        state      := writeReq
      }.otherwise {
        outputData := nextWord
        inputRow   := inputRow + 1.U
        state      := inputReq
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
        when(inputRow === iterReg - 1.U) {
          state := complete
        }.otherwise {
          inputRow := inputRow + 1.U
          when(outputColumn === outputWidth - 1.U) {
            outputRow    := outputRow + outputStride - outputWidth + 1.U
            outputColumn := 0.U
          }.otherwise {
            outputRow    := outputRow + 1.U
            outputColumn := outputColumn + 1.U
          }
          state    := inputReq
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
