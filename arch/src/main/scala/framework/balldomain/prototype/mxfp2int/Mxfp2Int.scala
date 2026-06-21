package framework.balldomain.prototype.mxfp2int

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.blink.mmio.MmioRead
import framework.top.GlobalConfig
import framework.balldomain.prototype.mxfp2int.configs.Mxfp2IntBallParam

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
 *   Read MMIO byte at rel_addr = block_idx with meta_bank = wr_bank.
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
class PipelinedMxfp2Int(val b: GlobalConfig) extends Module {
  val ballConfig = Mxfp2IntBallParam()
  val decoder    = MxfpDecoder.fromFormat(ballConfig.mxfpFormat)
  val elemBits   = decoder.elemBits

  val bankWidth = b.memDomain.bankWidth
  require(bankWidth == 128, s"Mxfp2IntBall requires bankWidth = 128, got $bankWidth")

  val wordsPerBlock = ballConfig.mxfpFormat match {
    case "MXFP4" => 1
    case "MXFP6" => 2 // padded: 192 bits data + 64 bits padding = 256 bits = 2 words
    case "MXFP8" => 2
    case _       => throw new IllegalArgumentException(s"Unsupported mxfpFormat: ${ballConfig.mxfpFormat}")
  }

  val FP_PER_BLOCK        = 32
  val INT8_PER_WORD       = bankWidth / 8                // 16
  val OUT_WORDS_PER_BLOCK = FP_PER_BLOCK / INT8_PER_WORD // 2

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "Mxfp2IntBall")
    .getOrElse(throw new IllegalArgumentException("Mxfp2IntBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
    val mmioRead  = Flipped(new MmioRead(b))
  })

  // ROB bookkeeping
  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  when(io.cmdReq.fire) {
    rob_id_reg     := io.cmdReq.bits.rob_id
    is_sub_reg     := io.cmdReq.bits.is_sub
    sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
  }

  // FSM
  val idle :: sReadReq :: sReadResp :: sMmioReq :: sMmioResp :: sWriteLo :: sWriteHi :: sComplete :: Nil = Enum(8)
  val state                                                                                              = RegInit(idle)

  // Per-instruction registers
  val raddr_reg     = RegInit(0.U(b.frontend.iter_len.W))
  val waddr_reg     = RegInit(0.U(b.frontend.iter_len.W))
  val rbank_reg     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg      = RegInit(0.U(b.frontend.iter_len.W))
  val block_idx_reg = RegInit(0.U(b.frontend.iter_len.W))
  val word_idx_reg  = RegInit(0.U(2.W)) // 0..wordsPerBlock-1

  val inputWordsReg = RegInit(VecInit(Seq.fill(2)(0.U(bankWidth.W)))) // max 2 words for MXFP6/8
  val scaleReg      = RegInit(0.U(8.W))

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
      Cat(inputWordsReg(wordIdx + 1)(highBits - 1, 0), inputWordsReg(wordIdx)(bankWidth - 1, bitInWord))
    } else {
      0.U(elemBits.W)
    }
  })

  // Combinationally compute all 32 INT8 values from the latched elements + scale
  val int8Vec = VecInit(fpElements.map(raw => decoder.dequant(raw, scaleReg).asUInt))

  val outWordLo = Cat((0 until INT8_PER_WORD).map(i => int8Vec(i)).reverse)
  val outWordHi = Cat((INT8_PER_WORD until FP_PER_BLOCK).map(i => int8Vec(i)).reverse)

  // === Default IO ===
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).rob_id           := rob_id_reg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id         := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.resp.ready    := false.B
    io.bankWrite(i).rob_id           := rob_id_reg
    io.bankWrite(i).ball_id          := 0.U
    io.bankWrite(i).bank_id          := wbank_reg
    io.bankWrite(i).group_id         := 0.U
  }

  io.mmioRead.req.valid         := false.B
  io.mmioRead.req.bits.rel_addr := 0.U
  io.mmioRead.resp.ready        := false.B
  io.mmioRead.meta_bank         := wbank_reg
  io.mmioRead.rob_id            := rob_id_reg
  io.mmioRead.ball_id           := 0.U
  io.cmdReq.ready               := state === idle
  io.cmdResp.valid              := false.B
  io.cmdResp.bits.rob_id        := rob_id_reg
  io.cmdResp.bits.is_sub        := is_sub_reg
  io.cmdResp.bits.sub_rob_id    := sub_rob_id_reg

  // === FSM ===
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        rbank_reg     := io.cmdReq.bits.cmd.op1_bank
        wbank_reg     := io.cmdReq.bits.cmd.wr_bank
        iter_reg      := io.cmdReq.bits.cmd.iter
        block_idx_reg := 0.U
        word_idx_reg  := 0.U
        raddr_reg     := 0.U
        waddr_reg     := 0.U
        when(io.cmdReq.bits.cmd.iter === 0.U) {
          state := sComplete
        }.otherwise {
          state := sReadReq
        }
      }
    }

    is(sReadReq) {
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := raddr_reg + block_idx_reg * wordsPerBlock.U + word_idx_reg
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
      io.mmioRead.req.valid         := true.B
      io.mmioRead.req.bits.rel_addr := block_idx_reg
      when(io.mmioRead.req.fire) {
        state := sMmioResp
      }
    }

    is(sMmioResp) {
      io.mmioRead.resp.ready := true.B
      when(io.mmioRead.resp.fire) {
        scaleReg := io.mmioRead.resp.bits.data(7, 0)
        state    := sWriteLo
      }
    }

    is(sWriteLo) {
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := waddr_reg + (block_idx_reg << 1)
      io.bankWrite(0).io.req.bits.data := outWordLo
      io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))
      io.bankWrite(0).io.resp.ready    := true.B
      when(io.bankWrite(0).io.req.fire) {
        state := sWriteHi
      }
    }

    is(sWriteHi) {
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := waddr_reg + (block_idx_reg << 1) + 1.U
      io.bankWrite(0).io.req.bits.data := outWordHi
      io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))
      io.bankWrite(0).io.resp.ready    := true.B
      when(io.bankWrite(0).io.req.fire) {
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
