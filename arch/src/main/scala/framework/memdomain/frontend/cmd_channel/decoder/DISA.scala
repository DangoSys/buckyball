package framework.memdomain.frontend.cmd_channel.decoder

import chisel3._
import chisel3.util._

object DISA {

  val MSET_BITPAT  = BitPat("b0010111") // 23
  val MVIN_BITPAT  = BitPat("b0011000") // 24
  val MVOUT_BITPAT = BitPat("b0011001") // 25
}
