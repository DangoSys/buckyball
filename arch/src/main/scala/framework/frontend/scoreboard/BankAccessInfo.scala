package framework.frontend.scoreboard

import chisel3._

//====----------------------------------------------------------====//
//  Bank access information extracted from instruction encoding.
//  It's instruction-agnostic: the scoreboard only sees read/write bank_id.
//====----------------------------------------------------------====//
class BankAccessInfo(val bankIdLen: Int) extends Bundle {
  val rd_bank_0_valid = Bool()
  val rd_bank_0_id    = UInt(bankIdLen.W)
  val rd_bank_1_valid = Bool()
  val rd_bank_1_id    = UInt(bankIdLen.W)
  val wr_bank_valid   = Bool()
  val wr_bank_id      = UInt(bankIdLen.W)
}

object BankAccessInfo {

  def none(bankIdLen: Int): BankAccessInfo = {
    val w = Wire(new BankAccessInfo(bankIdLen))
    w.rd_bank_0_valid := false.B
    w.rd_bank_0_id    := 0.U
    w.rd_bank_1_valid := false.B
    w.rd_bank_1_id    := 0.U
    w.wr_bank_valid   := false.B
    w.wr_bank_id      := 0.U
    w
  }

}
