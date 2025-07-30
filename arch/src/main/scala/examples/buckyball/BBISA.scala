package buckyball

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.buckyball.RoCCCommandBB

class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommandBB
}

object BBISA {
  val MVIN_BITPAT          = BitPat("b0011000")
  val MVOUT_BITPAT         = BitPat("b0011001")
  val MATMUL_WARP16_BITPAT = BitPat("b0100000") //32
  val BB_BBFP_MUL          = BitPat("b0011010") //26
  val MATMUL_WS            = BitPat("b0011011") //27
  val FENCE                = BitPat("b0011111") //31
}
  