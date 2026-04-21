package framework.balldomain.prototype.mxfp

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.mxfp.configs.MxfpBallParam

@instantiable
class PipelinedMxfp(val b: GlobalConfig) extends Module {
  val ballConfig = MxfpBallParam()
  val InputNum   = ballConfig.InputNum
  val inputWidth = ballConfig.inputWidth
  val bankWidth  = b.memDomain.bankWidth

  require(InputNum == 16, s"MxfpBall v1 requires InputNum = 16, got $InputNum")
  require(inputWidth == 32, s"MxfpBall v1 requires inputWidth = 32, got $inputWidth")
  require(bankWidth % inputWidth == 0, s"bankWidth must be divisible by inputWidth, got bankWidth=$bankWidth inputWidth=$inputWidth")
  require(bankWidth >= 96, s"MxfpBall requires bankWidth >= 96, got $bankWidth")

  val elemsPerWord = bankWidth / inputWidth
  require(InputNum % elemsPerWord == 0, s"InputNum must be divisible by elemsPerWord, got InputNum=$InputNum elemsPerWord=$elemsPerWord")
  val wordsPerBlock = InputNum / elemsPerWord

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "MxfpBall")
    .getOrElse(throw new IllegalArgumentException("MxfpBall not found in config"))
  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  // ---------------------------------------------------------------------------
  // FP32 helper functions
  // ---------------------------------------------------------------------------

  private def fpSign(x: UInt): UInt = x(31)
  private def fpExp(x: UInt): UInt  = x(30, 23)
  private def fpFrac(x: UInt): UInt = x(22, 0)

  private def isZero(x: UInt): Bool      = fpExp(x) === 0.U && fpFrac(x) === 0.U
  private def isSubnormal(x: UInt): Bool = fpExp(x) === 0.U && fpFrac(x) =/= 0.U
  private def isSpecial(x: UInt): Bool   = fpExp(x) === "hff".U

  private def normalExpOrZero(x: UInt): UInt =
    Mux(isZero(x) || isSubnormal(x) || isSpecial(x), 0.U(8.W), fpExp(x))

  // v1 approximation:
  // 4-bit magnitude under shared exponent
  private def quantizeMag4(x: UInt, sharedExp: UInt): UInt = {
    val exp  = fpExp(x)
    val frac = fpFrac(x)

    val sig24   = Cat(1.U(1.W), frac) // 24 bits
    val shiftAmt = (20.U(8.W) + sharedExp - exp)(5, 0)

    val shifted = sig24 >> shiftAmt
    val mag     = Wire(UInt(4.W))
    mag := 0.U

    when(isZero(x) || isSubnormal(x)) {
      mag := 0.U
    }.elsewhen(isSpecial(x)) {
      mag := 15.U
    }.elsewhen(exp > sharedExp) {
      mag := 15.U
    }.otherwise {
      mag := Mux(shifted >= 15.U, 15.U, shifted(3, 0))
    }

    mag
  }

  // Pack 16 FP32 values into one MX6 block
  //
  // Layout (LSB first):
  //   [7:0]    : global exponent
  //   [15:8]   : 8 micro bits
  //   [95:16]  : 16 * (sign + 4-bit mag) = 80 bits
  //   [bankWidth-1:96] : zero
  private def packMx6Block(elems: Seq[UInt]): UInt = {
    require(elems.length == InputNum, s"packMx6Block expects $InputNum elements, got ${elems.length}")

    val exps      = elems.map(normalExpOrZero)
    val globalExp = exps.reduce((a, b) => Mux(a > b, a, b))

    val microBits = (0 until InputNum / 2).map { p =>
      val e0      = exps(2 * p)
      val e1      = exps(2 * p + 1)
      val pairMax = Mux(e0 > e1, e0, e1)
      (globalExp =/= 0.U) && (pairMax + 1.U <= globalExp)
    }

    val elemPayloads = (0 until InputNum).map { i =>
      val pairIdx  = i / 2
      val localExp = Mux(microBits(pairIdx), globalExp - 1.U, globalExp)
      val signBit  = fpSign(elems(i))
      val mag4     = quantizeMag4(elems(i), localExp)
      Cat(signBit, mag4) // 5 bits
    }

    val microPacked = Cat(microBits.reverse.map(_.asUInt)) // 8 bits
    val elemPacked  = Cat(elemPayloads.reverse)            // 80 bits
    val packed96    = Cat(elemPacked, microPacked, globalExp)

    if (bankWidth > 96) {
      Cat(0.U((bankWidth - 96).W), packed96)
    } else {
      packed96(bankWidth - 1, 0)
    }
  }

  // ---------------------------------------------------------------------------
  // ROB bookkeeping
  // ---------------------------------------------------------------------------

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

  // ---------------------------------------------------------------------------
  // State machine
  // iter is interpreted as number of MX blocks
  // Each block consumes wordsPerBlock input words and produces 1 output word
  // ---------------------------------------------------------------------------

  val idle :: sRead :: sPack :: sWrite :: complete :: Nil = Enum(5)
  val state                                               = RegInit(idle)

  val readCntWidth = log2Ceil(wordsPerBlock + 1)

  val fp32Buf        = RegInit(VecInit(Seq.fill(InputNum)(0.U(inputWidth.W))))
  val packedBlockReg = RegInit(0.U(bankWidth.W))

  val readReqCounter  = RegInit(0.U(readCntWidth.W))
  val readRespCounter = RegInit(0.U(readCntWidth.W))

  val raddr_reg        = RegInit(0.U(b.frontend.iter_len.W))
  val waddr_reg        = RegInit(0.U(b.frontend.iter_len.W))
  val rbank_reg        = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg        = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val remainingBlocks  = RegInit(0.U(b.frontend.iter_len.W))

  val writeMaskReg = RegInit(VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W))))

  // ---------------------------------------------------------------------------
  // Default IO
  // ---------------------------------------------------------------------------

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).bank_id  := rbank_reg
    io.bankRead(i).group_id := 0.U
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).bank_id  := wbank_reg
    io.bankWrite(i).group_id := 0.U
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  // ---------------------------------------------------------------------------
  // Main FSM
  // ---------------------------------------------------------------------------

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        readReqCounter  := 0.U
        readRespCounter := 0.U
        raddr_reg       := 0.U
        waddr_reg       := 0.U
        rbank_reg       := io.cmdReq.bits.cmd.op1_bank
        wbank_reg       := io.cmdReq.bits.cmd.wr_bank
        remainingBlocks := io.cmdReq.bits.cmd.iter

        for (i <- 0 until b.memDomain.bankMaskLen) {
          writeMaskReg(i) := 1.U
        }

        when(io.cmdReq.bits.cmd.iter === 0.U) {
          state := complete
        }.otherwise {
          state := sRead
        }
      }
    }

    is(sRead) {
      io.bankRead(0).io.resp.ready := true.B

      io.bankRead(0).io.req.valid     := readReqCounter < wordsPerBlock.U
      io.bankRead(0).io.req.bits.addr := raddr_reg + readReqCounter

      when(io.bankRead(0).io.req.fire) {
        readReqCounter := readReqCounter + 1.U
      }

      when(io.bankRead(0).io.resp.fire) {
        val dataWord = io.bankRead(0).io.resp.bits.data

        for (w <- 0 until wordsPerBlock) {
          when(readRespCounter === w.U) {
            for (i <- 0 until elemsPerWord) {
              val hi = (i + 1) * inputWidth - 1
              val lo = i * inputWidth
              fp32Buf(w * elemsPerWord + i) := dataWord(hi, lo)
            }
          }
        }

        when(readRespCounter === (wordsPerBlock - 1).U) {
          state := sPack
        }

        readRespCounter := readRespCounter + 1.U
      }
    }

    is(sPack) {
      packedBlockReg := packMx6Block((0 until InputNum).map(i => fp32Buf(i)))
      state := sWrite
    }

    is(sWrite) {
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := waddr_reg
      io.bankWrite(0).io.req.bits.data := packedBlockReg
      io.bankWrite(0).io.req.bits.mask := writeMaskReg
      io.bankWrite(0).io.resp.ready    := true.B

      when(io.bankWrite(0).io.req.fire) {
        when(remainingBlocks > 1.U) {
          remainingBlocks := remainingBlocks - 1.U
          raddr_reg       := raddr_reg + wordsPerBlock.U
          waddr_reg       := waddr_reg + 1.U
          readReqCounter  := 0.U
          readRespCounter := 0.U
          state           := sRead
        }.otherwise {
          state := complete
        }
      }
    }

    is(complete) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := state === idle
  io.status.running := (state === sRead) || (state === sPack) || (state === sWrite)
}
