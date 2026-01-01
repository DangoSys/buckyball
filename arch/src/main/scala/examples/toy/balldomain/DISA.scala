package examples.toy.balldomain

import chisel3._
import chisel3.util._

object DISA {
  val BB_BBFP_MUL   = BitPat("b0011010") // 26
  val MATMUL_WS     = BitPat("b0011011") // 27
  val MATMUL_WARP16 = BitPat("b0100000") // 32
  val IM2COL        = BitPat("b0100001") // 33
  val TRANSPOSE     = BitPat("b0100010") // 34
  val RELU          = BitPat("b0100110") // 38
  val CONCAT        = BitPat("b0100111") // 39
  val NNLUT         = BitPat("b0101000") // 40
  val SNN           = BitPat("b0101001") // 41
  val ABFT_SYSTOLIC = BitPat("b0101010") // 42
  val CONV          = BitPat("b0101011") // 43
  val CIM           = BitPat("b0101100") // 44
  val TRANSFER      = BitPat("b0101101") // 45
}
