package framework.acc.memdomain

import chisel3._
import chisel3.util._

object MvoverISA {
  val Funct       = 0x0d
  val ThreadBits  = 8
  val BankBits    = 10
  val AddressBits = 16
}

/** Contiguous rows between logical bank endpoints attached to one fabric. */
class MvoverCommand extends Bundle {
  val sourceThread = UInt(MvoverISA.ThreadBits.W)
  val sourceBank   = UInt(MvoverISA.BankBits.W)
  val sourceAddr   = UInt(MvoverISA.AddressBits.W)
  val targetThread = UInt(MvoverISA.ThreadBits.W)
  val targetBank   = UInt(MvoverISA.BankBits.W)
  val targetAddr   = UInt(MvoverISA.AddressBits.W)
  val rows         = UInt((MvoverISA.AddressBits + 1).W)
}

class MvoverPort extends Bundle {
  val command    = Decoupled(new MvoverCommand)
  val completion = Flipped(Decoupled(Bool()))
}
