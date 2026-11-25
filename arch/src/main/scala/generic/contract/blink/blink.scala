package framework.generic.contract.blink

import chisel3._
import chisel3.util._

/**
 * What a Ball knows of its host: the widths of BLink and the geometry of a bank.
 * A host with these values hosts any Ball; nothing else of the host reaches a Ball.
 * Every bank a Ball is given has this geometry, whatever storage the host backs it with.
 */
case class BlinkParam(
  robIdBits:      Int,
  dataBits:       Int,
  iterBits:       Int,
  bankIdBits:     Int,
  groupIdBits:    Int,
  groupCountBits: Int,
  bankNum:        Int,
  bankEntries:    Int,
  bankWidth:      Int,
  bankMaskLen:    Int,
  mmioBytes:      Int,
  mmioDataBits:   Int)

/**
 * One Ball operation. Banks are already renamed; each operand covers `col` groups starting at `group`.
 */
class BallCommand(link: BlinkParam) extends Bundle {
  val rob_id       = UInt(link.robIdBits.W)
  val funct        = UInt(7.W)
  val rs2          = UInt(link.dataBits.W)
  val iter         = UInt(link.iterBits.W)
  val op1_bank     = UInt(link.bankIdBits.W)
  val op2_bank     = UInt(link.bankIdBits.W)
  val wr_bank      = UInt(link.bankIdBits.W)
  val op1_group    = UInt(link.groupIdBits.W)
  val op2_group    = UInt(link.groupIdBits.W)
  val wr_group     = UInt(link.groupIdBits.W)
  val op1_col      = UInt(link.groupCountBits.W)
  val op2_col      = UInt(link.groupCountBits.W)
  val wr_col       = UInt(link.groupCountBits.W)
  // Renamed bank holding the previous destination value for accumulation.
  val wr_read_bank = UInt(link.bankIdBits.W)
  // Destination bank before renaming; identifies chained writes to one architectural bank.
  val wr_arch_bank = UInt(link.bankIdBits.W)
}

class BallResponse(link: BlinkParam) extends Bundle {
  val rob_id = UInt(link.robIdBits.W)
}
