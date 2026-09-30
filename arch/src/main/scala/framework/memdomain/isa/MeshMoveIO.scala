package framework.memdomain.isa

import chisel3._
import chisel3.util._

object MeshMoveISA {
  val Funct       = 0x50
  val CoreBits    = 8
  val BankBits    = 10
  val AddressBits = 16
}

/** One full-width row between private Banks in the same Tile. */
class MeshMoveCommand extends Bundle {
  val sourceCore = UInt(MeshMoveISA.CoreBits.W)
  val sourceBank = UInt(MeshMoveISA.BankBits.W)
  val sourceAddr = UInt(MeshMoveISA.AddressBits.W)
  val targetCore = UInt(MeshMoveISA.CoreBits.W)
  val targetBank = UInt(MeshMoveISA.BankBits.W)
  val targetAddr = UInt(MeshMoveISA.AddressBits.W)
}

class MeshMovePort extends Bundle {
  val command    = Decoupled(new MeshMoveCommand)
  val completion = Flipped(Decoupled(Bool()))
}
