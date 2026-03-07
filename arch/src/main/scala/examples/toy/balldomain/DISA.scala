package examples.toy.balldomain

import chisel3._
import chisel3.util._

object DISA {
  val MATMUL_WARP16 = BitPat("b0100000") // 32
  val IM2COL        = BitPat("b0100001") // 33
  val TRANSPOSE     = BitPat("b0100010") // 34
  val RELU          = BitPat("b0100110") // 38
  val SYSTOLIC      = BitPat("b0100111") // 39
  val QUANT         = BitPat("b0101000") // 40
  val DEQUANT       = BitPat("b0101001") // 41

  // Gemmini systolic array instructions
  val GEMMINI_CONFIG              = BitPat("b0101010") // 42
  val GEMMINI_PRELOAD             = BitPat("b0101011") // 43
  val GEMMINI_COMPUTE_PRELOADED   = BitPat("b0101100") // 44
  val GEMMINI_COMPUTE_ACCUMULATED = BitPat("b0101101") // 45
  val GEMMINI_FLUSH               = BitPat("b0101110") // 46
}
