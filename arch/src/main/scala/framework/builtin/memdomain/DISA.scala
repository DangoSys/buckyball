package framework.builtin.memdomain

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
// import framework.ballcore.ballcore.RoCCCommandBB
import freechips.rocketchip.tile._  


class BuckyBallRawCmd(implicit p: Parameters) extends Bundle {
  val cmd = new RoCCCommand
}

object DISA {
  val MVIN_BITPAT          = BitPat("b0011000") // 24
  val MVOUT_BITPAT         = BitPat("b0011001") // 25
}