package examples.balls.toint8

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.toint8.{Params => ParametersProto}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import hardfloat.{recFNFromFN, INToRecFN, RecFNToIN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}
import framework.lib.arith.Fma

import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class ToInt8Ball(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  val f32Funct = ball.instructions
    .find(_.mnemonic == "QUANT_F32_TO_I8").get.funct7

  val i32Funct = ball.instructions
    .find(_.mnemonic == "QUANT_I32_TO_I8").get.funct7

  require(mp.bankWidth == 128, "ToInt8Ball requires 128-bit bank rows")
  require(
    mp.bankEntries >= 4,
    "ToInt8Ball requires at least four bank rows"
  )
  require(ball.inBW == 1, "ToInt8Ball requires inBW=1")
  require(ball.outBW == 1, "ToInt8Ball requires outBW=1")
  require((f32Funct >> 4) == 3, "QUANT_F32_TO_I8 must encode one read and one write")
  require((i32Funct >> 4) == 4, "QUANT_I32_TO_I8 must encode two reads and one write")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: scaleReq :: scaleResp :: inputReq :: inputResp :: pack :: packDrain :: writeReq :: writeResp :: complete :: Nil =
    Enum(10)
  val state                                                                                                                   = RegInit(idle)

  val robIdReg     = RegInit(0.U(log2Up(fp.rob_entries).W))
  val inputBank    = RegInit(0.U(fp.bank_id_len.W))
  val scaleBank    = RegInit(0.U(fp.bank_id_len.W))
  val outputBank   = RegInit(0.U(fp.bank_id_len.W))
  val bases        = Reg(Vec(3, UInt(mp.groupIdWidth.W)))
  // Every group of a virtual bank is one 16-channel block with the same rows.
  val groups       = RegInit(0.U(log2Up(mp.bankNum + 1).W))
  val group        = RegInit(0.U(log2Up(mp.bankNum + 1).W))
  val outputBase   = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val iterReg      = RegInit(0.U(fp.iter_len.W))
  val inputBaseReg = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  // Relative row within the command.  inputBaseReg is the absolute bank row.
  val relativeRow  = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val outputRow    = RegInit(0.U(log2Ceil(mp.bankEntries).W))
  val outputWidth  = RegInit(0.U(7.W))
  val outputStride = RegInit(0.U(7.W))
  val outputColumn = RegInit(0.U(7.W))
  val scaleRow     = RegInit(0.U(2.W))
  val i32Mode      = RegInit(false.B)
  val reluReg      = RegInit(false.B)
  val tensorScale  = RegInit(0.U(32.W))
  val scales       = Reg(Vec(16, UInt(32.W)))
  val inputData    = Reg(UInt(128.W))
  val outputData   = RegInit(0.U(128.W))
  val writeData    = Reg(UInt(128.W))
  val lane         = RegInit(0.U(2.W))
  val outputBytes  = Reg(Vec(4, UInt(8.W)))

  def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  val intToFloat = Module(new INToRecFN(32, 8, 24))

  // The unpipelined recoded-float multiplier was the critical Buckyball path
  // (about 8.5 ns before clock uncertainty).  FMA mode with c=0 preserves the
  // multiplication result while splitting the datapath into three stages.
  val multiply = Instantiate(new Fma(3, 8, 24))

  val floatToInt = Module(new RecFNToIN(8, 24, 8))
  val input      = (inputData >> (lane << 5))(31, 0)
  val scale      = Mux(i32Mode, scales(Cat(relativeRow(1, 0), lane)), tensorScale)
  val packIssue  = state === pack

  val inputStage0  = RegEnable(input, packIssue)
  val scaleStage0  = RegEnable(scale, packIssue)
  val laneStage0   = RegEnable(lane, packIssue)
  val convertValid = RegNext(packIssue, false.B)

  val multiplyInput = RegEnable(
    Mux(i32Mode, intToFloat.io.out, recFNFromFN(8, 24, inputStage0)),
    convertValid
  )

  val multiplyScale = RegEnable(recFNFromFN(8, 24, scaleStage0), convertValid)
  val laneStage1    = RegEnable(laneStage0, convertValid)
  val multiplyValid = RegNext(convertValid, false.B)

  val multiplyResult = RegEnable(multiply.io.out, multiply.io.validout)
  val multiplyLane   = Pipe(multiplyValid, laneStage1, 3)
  val laneStage2     = RegEnable(multiplyLane.bits, multiply.io.validout)
  val toIntValid     = RegNext(multiply.io.validout, false.B)

  val quantizedResult = RegEnable(
    Mux(reluReg && floatToInt.io.out(7), 0.U, floatToInt.io.out),
    toIntValid
  )

  val quantizedLane  = RegEnable(laneStage2, toIntValid)
  val quantizedValid = RegNext(toIntValid, false.B)

  intToFloat.io.signedIn       := true.B
  intToFloat.io.in             := inputStage0
  intToFloat.io.roundingMode   := round_near_even
  intToFloat.io.detectTininess := tininess_afterRounding

  multiply.io.a              := multiplyInput
  multiply.io.b              := multiplyScale
  multiply.io.c              := recFNFromFN(8, 24, 0.U)
  multiply.io.op             := 0.U
  multiply.io.validin        := multiplyValid
  multiply.io.roundingMode   := round_near_even
  multiply.io.detectTininess := tininess_afterRounding.asUInt

  floatToInt.io.in           := multiplyResult
  floatToInt.io.roundingMode := round_near_even
  floatToInt.io.signedOut    := true.B

  blink.cmdReq.ready        := state === idle
  blink.cmdResp.valid       := state === complete
  blink.cmdResp.bits.rob_id := robIdReg
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  blink.bankRead(0).rob_id           := robIdReg
  blink.bankRead(0).bank_id          := Mux(state === scaleReq || state === scaleResp, scaleBank, inputBank)
  blink.bankRead(0).group_id         := Mux(state === scaleReq || state === scaleResp, bases(1), bases(0)) + group
  blink.bankRead(0).io.req.valid     := false.B
  blink.bankRead(0).io.req.bits.addr := 0.U
  blink.bankRead(0).io.resp.ready    := false.B

  blink.bankWrite(0).rob_id           := robIdReg
  blink.bankWrite(0).bank_id          := outputBank
  blink.bankWrite(0).group_id         := bases(2) + group
  blink.bankWrite(0).io.req.valid     := false.B
  blink.bankWrite(0).io.req.bits.addr := outputRow
  blink.bankWrite(0).io.req.bits.data := writeData
  blink.bankWrite(0).io.req.bits.mask := VecInit(
    Seq.fill(mp.bankMaskLen)(true.B)
  )
  blink.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd   = blink.cmdReq.bits
        val isF32 = cmd.funct === f32Funct.U
        val isI32 = cmd.funct === i32Funct.U

        assert(isF32 || isI32, "ToInt8Ball received an unknown funct7")
        assert(cmd.iter > 0.U && cmd.iter(1, 0) === 0.U, "ToInt8Ball iter must be a positive multiple of four")
        assert(
          cmd.iter <= mp.bankEntries.U,
          "ToInt8Ball input exceeds bank depth"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op1_col === cmd.wr_col && cmd.op1_col <= mp.bankNum.U,
          "ToInt8Ball input and output must have the same channel groups"
        )
        assert(
          cmd.op1_bank =/= cmd.wr_bank,
          "ToInt8Ball input and output banks must differ"
        )
        when(isF32) {
          assert(cmd.rs2(63, 32) === 0.U, "QUANT_F32_TO_I8 reserves rs2[63:32]")
          assert(positiveFinite(cmd.rs2(31, 0)), "QUANT_F32_TO_I8 scale must be finite and positive")
        }.otherwise {
          assert(cmd.op2_col === cmd.op1_col, "QUANT_I32_TO_I8 scale must have one group per channel group")
          assert(
            cmd.op1_bank =/= cmd.op2_bank,
            "QUANT_I32_TO_I8 input and scale banks must differ"
          )
          assert(
            cmd.op2_bank =/= cmd.wr_bank,
            "QUANT_I32_TO_I8 scale and output banks must differ"
          )
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
          assert(
            outputEnd <= mp.bankEntries.U,
            "QUANT_I32_TO_I8 output exceeds bank depth"
          )
          assert(
            inputBase +& cmd.iter <= mp.bankEntries.U,
            "QUANT_I32_TO_I8 input exceeds bank depth"
          )
        }

        robIdReg     := blink.cmdReq.bits.rob_id
        inputBank    := cmd.op1_bank
        scaleBank    := cmd.op2_bank
        outputBank   := cmd.wr_bank
        bases        := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        groups       := cmd.op1_col
        group        := 0.U
        outputBase   := Mux(isI32, cmd.rs2(7, 1), 0.U)
        iterReg      := cmd.iter
        inputBaseReg := Mux(isI32, cmd.rs2(34, 29), 0.U)
        relativeRow  := 0.U
        outputRow    := Mux(isI32, cmd.rs2(7, 1), 0.U)
        outputWidth  := Mux(isI32, cmd.rs2(14, 8), 0.U)
        outputStride := Mux(isI32, cmd.rs2(28, 22), 0.U)
        outputColumn := 0.U
        scaleRow     := 0.U
        i32Mode      := isI32
        reluReg      := isI32 && cmd.rs2(0)
        tensorScale  := cmd.rs2(31, 0)
        outputData   := 0.U
        lane         := 0.U
        state        := Mux(isI32, scaleReq, inputReq)
      }
    }

    is(scaleReq) {
      blink.bankRead(0).io.req.valid     := true.B
      blink.bankRead(0).io.req.bits.addr := scaleRow
      when(blink.bankRead(0).io.req.fire) {
        state := scaleResp
      }
    }

    is(scaleResp) {
      blink.bankRead(0).io.resp.ready := true.B
      when(blink.bankRead(0).io.resp.fire) {
        for (lane <- 0 until 4) {
          val value = blink.bankRead(0).io.resp.bits.data(32 * lane + 31, 32 * lane)
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
      blink.bankRead(0).io.req.valid     := true.B
      blink.bankRead(0).io.req.bits.addr := inputBaseReg +& relativeRow
      when(blink.bankRead(0).io.req.fire) {
        state := inputResp
      }
    }

    is(inputResp) {
      blink.bankRead(0).io.resp.ready := true.B
      when(blink.bankRead(0).io.resp.fire) {
        inputData := blink.bankRead(0).io.resp.bits.data
        lane      := 0.U
        state     := pack
      }
    }

    is(pack) {
      when(!i32Mode) {
        assert(input(30, 23) =/= 255.U, "QUANT_F32_TO_I8 input must be finite")
      }
      when(lane === 3.U) {
        state := packDrain
      }.otherwise {
        lane := lane + 1.U
      }
    }

    is(packDrain) {}

    is(writeReq) {
      blink.bankWrite(0).io.req.valid := true.B
      when(blink.bankWrite(0).io.req.fire) {
        state := writeResp
      }
    }

    is(writeResp) {
      blink.bankWrite(0).io.resp.ready := true.B
      when(blink.bankWrite(0).io.resp.fire) {
        when(relativeRow === iterReg - 1.U) {
          when(group + 1.U === groups) {
            state := complete
          }.otherwise {
            group        := group + 1.U
            relativeRow  := 0.U
            outputRow    := outputBase
            outputColumn := 0.U
            scaleRow     := 0.U
            state        := Mux(i32Mode, scaleReq, inputReq)
          }
        }.otherwise {
          relativeRow := relativeRow + 1.U
          when(outputColumn === outputWidth - 1.U) {
            outputRow    := outputRow + outputStride - outputWidth + 1.U
            outputColumn := 0.U
          }.otherwise {
            outputRow    := outputRow + 1.U
            outputColumn := outputColumn + 1.U
          }
          state       := inputReq
        }
      }
    }

    is(complete) {
      when(blink.cmdResp.fire) {
        state := idle
      }
    }
  }

  when((state === pack || state === packDrain) && quantizedValid) {
    outputBytes(quantizedLane) := quantizedResult
    when(quantizedLane === 3.U) {
      val bytes    = Cat(quantizedResult, outputBytes(2), outputBytes(1), outputBytes(0))
      val nextWord = outputData | (bytes << (relativeRow(1, 0) << 5))
      when(relativeRow(1, 0) === 3.U) {
        writeData  := nextWord
        outputData := 0.U
        state      := writeReq
      }.otherwise {
        outputData  := nextWord
        relativeRow := relativeRow + 1.U
        state       := inputReq
      }
    }
  }
}

object ToInt8Ball extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new ToInt8Ball(fp, mp, bp, id)).blink, None)

}
