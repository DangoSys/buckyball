package examples.toy.balldomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import framework.rocket.RoCCCommandBB
import freechips.rocketchip.tile._


class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommandBB
}

object DISA {
  val BB_BBFP_MUL          = BitPat("b0011010") // 26
  val MATMUL_WS            = BitPat("b0011011") // 27
  val MATMUL_WARP16_BITPAT = BitPat("b0100000") // 32
  val IM2COL               = BitPat("b0100001") // 33
  val TRANSPOSE            = BitPat("b0100010") // 34
  val GELU                 = BitPat("b0100011") // 35
  val LAYERNORM            = BitPat("b0100100") // 36
  val SOFTMAX              = BitPat("b0100101") // 37
  val RELU                 = BitPat("b0100110") // 38
}
