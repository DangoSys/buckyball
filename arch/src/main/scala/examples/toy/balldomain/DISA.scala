package examples.toy.balldomain

import chisel3._
import chisel3.util._

object DISA {
  val MATMUL_WARP16 = BitPat("b0100000") // 32
  val IM2COL        = BitPat("b0100001") // 33
  val TRANSPOSE     = BitPat("b0100010") // 34
  val RELU          = BitPat("b0100110") // 38
  val SYSTOLIC      = BitPat("b0100111") // 39
}
