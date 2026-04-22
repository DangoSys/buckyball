package framework.balldomain.decoder

import chisel3._
import chisel3.util._

/**
 * Wire-level command bundle exchanged between user-defined Ball decoders
 * and framework infrastructure (BallReservationStation, BBus). The
 * concrete decode logic (funct7 → BID/operands) lives in user code.
 */
class BallDecodeCmd(numBanks: Int, iterLen: Int) extends Bundle {
  val bid    = UInt(5.W)
  val funct7 = UInt(7.W)
  val iter   = UInt(iterLen.W)

  val op1_en        = Bool()
  val op2_en        = Bool()
  val wr_spad_en    = Bool()
  val op1_from_spad = Bool()
  val op2_from_spad = Bool()
  val special       = UInt(64.W)

  val op1_bank = UInt(log2Up(numBanks).W)
  val op2_bank = UInt(log2Up(numBanks).W)
  val wr_bank  = UInt(log2Up(numBanks).W)

  val rs1 = UInt(64.W)
  val rs2 = UInt(64.W)
}
