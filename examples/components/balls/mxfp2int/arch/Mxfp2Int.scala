package examples.balls.mxfp2int

import framework.acc.balldomain.configs.BallRouteParam

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}
import framework.acc.balldomain.mmio_bank.MmioRead
import examples.balls.mxfp2int.configs.Mxfp2IntBallParam

/**
 * PipelinedMxfp2Int - MXFP to INT8 dequantizer with MMIO per-block scale.
 *
 * Supports MXFP4, MXFP6, MXFP8 formats (32 elements per block).
 *
 * Input format (main bank, per block):
 *   MXFP4: 32 × 4-bit = 128 bits = 1 word
 *   MXFP6: 32 × 6-bit = 192 bits, padded to 2 words (256 bits, high 64 bits = 0)
 *   MXFP8: 32 × 8-bit = 256 bits = 2 words
 *
 * Scale (MMIO bank): per-block E8M0 (8-bit biased exponent, bias=127).
 *
 * FP4/6/8 decoding (v1, simplified): signed_val = raw - (1 << (bits-1))
 *   FP4: raw - 8  (range -8..+7)
 *   FP6: raw - 32 (range -32..+31)
 *   FP8: raw - 128 (range -128..+127)
 *
 * Output (per block, 2 SRAM words = 32 bytes):
 *   32 × INT8 = saturate(signed_val << (scale - 127)).
 *   Lower 16 INT8 to row (2 * block_idx), upper 16 to row (2 * block_idx + 1).
 *
 * iter encodes the number of MXFP blocks.
 */
@instantiable
class PipelinedMxfp2Int(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     Mxfp2IntBallParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val ballConfig      = ballParam
  val decoder         = MxfpDecoder.fromFormat(ballConfig.mxfpFormat)
  val elemBits        = decoder.elemBits

  val bankWidth = memParam.bankWidth
  require(
    bankWidth == 128,
    s"Mxfp2IntBall requires bankWidth = 128, got $bankWidth"
  )

  val wordsPerBlock = ballConfig.mxfpFormat match {
    case "MXFP4" => 1
    case "MXFP6" =>
      2 // padded: 192 bits data + 64 bits padding = 256 bits = 2 words
    case "MXFP8" => 2
    case _       =>
      throw new IllegalArgumentException(
        s"Unsupported mxfpFormat: ${ballConfig.mxfpFormat}"
      )
  }

  val FP_PER_BLOCK  = 32
  val INT8_PER_WORD = bankWidth / 8 // 16

  val ballMapping = instance

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      inBW,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val bankWrite = Vec(
      outBW,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val status   = new BallStatus
    val mmioRead = Flipped(new MmioRead(memParam))
  })

  // ROB bookkeeping
  val rob_id_reg = RegInit(0.U(log2Up(frontendParam.rob_entries).W))

  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  // FSM
  val idle :: sReadReq :: sReadResp :: sMmioReq :: sMmioResp :: sWriteLoReq :: sWriteLoResp :: sWriteHiReq :: sWriteHiResp :: sComplete :: Nil =
    Enum(10)
  val state                                                                                                                                    = RegInit(idle)

  // Per-instruction registers
  val raddr_reg     = RegInit(0.U(frontendParam.iter_len.W))
  val waddr_reg     = RegInit(0.U(frontendParam.iter_len.W))
  val rbank_reg     = RegInit(0.U(frontendParam.bank_id_len.W))
  val wbank_reg     = RegInit(0.U(frontendParam.bank_id_len.W))
  val bases         = Reg(Vec(3, UInt(memParam.groupIdWidth.W)))
  val iter_reg      = RegInit(0.U(frontendParam.iter_len.W))
  val block_idx_reg = RegInit(0.U(frontendParam.iter_len.W))
  val word_idx_reg  = RegInit(0.U(2.W)) // 0..wordsPerBlock-1

  val inputWordsReg = RegInit(
    VecInit(Seq.fill(2)(0.U(bankWidth.W)))
  ) // max 2 words for MXFP6/8
  val scaleReg = RegInit(0.U(8.W))

  // Extract 32 FP elements from inputWordsReg based on format
  val fpElements = VecInit(Seq.tabulate(FP_PER_BLOCK) { i =>
    val bitOffset = i * elemBits
    val wordIdx   = bitOffset / bankWidth
    val bitInWord = bitOffset % bankWidth
    if (wordIdx < wordsPerBlock && bitInWord + elemBits <= bankWidth) {
      inputWordsReg(wordIdx)(bitInWord + elemBits - 1, bitInWord)
    } else if (wordIdx < wordsPerBlock - 1 && bitInWord + elemBits > bankWidth) {
      // Cross-word boundary (only for MXFP6 if packed, but we use padded so this won't happen)
      val lowBits  = bankWidth - bitInWord
      val highBits = elemBits - lowBits
      Cat(
        inputWordsReg(wordIdx + 1)(highBits - 1, 0),
        inputWordsReg(wordIdx)(bankWidth - 1, bitInWord)
      )
    } else {
      0.U(elemBits.W)
    }
  })

  // Combinationally compute all 32 INT8 values from the latched elements + scale
  val int8Vec = VecInit(
    fpElements.map(raw => decoder.dequant(raw, scaleReg).asUInt)
  )

  val outWordLo = Cat((0 until INT8_PER_WORD).map(i => int8Vec(i)).reverse)

  val outWordHi = Cat(
    (INT8_PER_WORD until FP_PER_BLOCK).map(i => int8Vec(i)).reverse
  )

  // === Default IO ===
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).rob_id           := rob_id_reg
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id         := bases(0)
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(
      Seq.fill(memParam.bankMaskLen)(0.U(1.W))
    )
    io.bankWrite(i).io.resp.ready    := false.B
    io.bankWrite(i).rob_id           := rob_id_reg
    io.bankWrite(i).bank_id          := wbank_reg
    io.bankWrite(i).group_id         := bases(2)
  }

  io.mmioRead.req.valid     := false.B
  io.mmioRead.req.bits.addr := 0.U
  io.mmioRead.resp.ready    := false.B
  io.cmdReq.ready           := state === idle
  io.cmdResp.valid          := false.B
  io.cmdResp.bits.rob_id    := rob_id_reg

  // === FSM ===
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        rbank_reg     := io.cmdReq.bits.op1_bank
        bases         := VecInit(io.cmdReq.bits.op1_group, io.cmdReq.bits.op2_group, io.cmdReq.bits.wr_group)
        wbank_reg     := io.cmdReq.bits.wr_bank
        iter_reg      := io.cmdReq.bits.iter
        block_idx_reg := 0.U
        word_idx_reg  := 0.U
        raddr_reg     := 0.U
        waddr_reg     := 0.U
        when(io.cmdReq.bits.iter === 0.U) {
          state := sComplete
        }.otherwise {
          state := sReadReq
        }
      }
    }

    is(sReadReq) {
      io.bankRead(0).io.req.valid := true.B
      io.bankRead(0)
        .io
        .req
        .bits
        .addr                     := raddr_reg + block_idx_reg * wordsPerBlock.U + word_idx_reg
      when(io.bankRead(0).io.req.fire) {
        state := sReadResp
      }
    }

    is(sReadResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        inputWordsReg(word_idx_reg) := io.bankRead(0).io.resp.bits.data
        when(word_idx_reg === (wordsPerBlock - 1).U) {
          word_idx_reg := 0.U
          state        := sMmioReq
        }.otherwise {
          word_idx_reg := word_idx_reg + 1.U
          state        := sReadReq
        }
      }
    }

    is(sMmioReq) {
      io.mmioRead.req.valid     := true.B
      io.mmioRead.req.bits.addr := block_idx_reg
      when(io.mmioRead.req.fire) {
        state := sMmioResp
      }
    }

    is(sMmioResp) {
      io.mmioRead.resp.ready := true.B
      when(io.mmioRead.resp.fire) {
        scaleReg := io.mmioRead.resp.bits.data(7, 0)
        state    := sWriteLoReq
      }
    }

    is(sWriteLoReq) {
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := waddr_reg + (block_idx_reg << 1)
      io.bankWrite(0).io.req.bits.data := outWordLo
      io.bankWrite(0).io.req.bits.mask := VecInit(
        Seq.fill(memParam.bankMaskLen)(1.U(1.W))
      )
      when(io.bankWrite(0).io.req.fire) {
        state := sWriteLoResp
      }
    }

    is(sWriteLoResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        state := sWriteHiReq
      }
    }

    is(sWriteHiReq) {
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := waddr_reg + (block_idx_reg << 1) + 1.U
      io.bankWrite(0).io.req.bits.data := outWordHi
      io.bankWrite(0).io.req.bits.mask := VecInit(
        Seq.fill(memParam.bankMaskLen)(1.U(1.W))
      )
      when(io.bankWrite(0).io.req.fire) {
        state := sWriteHiResp
      }
    }

    is(sWriteHiResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(block_idx_reg === iter_reg - 1.U) {
          state := sComplete
        }.otherwise {
          block_idx_reg := block_idx_reg + 1.U
          state         := sReadReq
        }
      }
    }

    is(sComplete) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := state === idle
  io.status.running := state =/= idle
}
