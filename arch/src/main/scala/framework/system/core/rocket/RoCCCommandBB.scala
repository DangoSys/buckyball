package framework.system.core.rocket

import chisel3._
import chisel3.util._

/** RoCC response bundle */
class RoCCResponseBB(xLen: Int = 64) extends Bundle {
  val rd   = Bits(5.W)
  val data = Bits(xLen.W)
}

/** RoCC interface between a core and an accelerator. */
class RoCCIO(xLen: Int = 64) extends Bundle {
  val cmd       = Flipped(Decoupled(new RoCCCommandBB(xLen)))
  val resp      = Decoupled(new RoCCResponseBB(xLen))
  val busy      = Output(Bool())
  val interrupt = Output(Bool())
  val exception = Input(Bool())
}

/** RoCC command bundle */
class RoCCCommandBB(xLen: Int = 64) extends Bundle {
  val raw_inst = UInt(32.W)
  val pc       = UInt(xLen.W)
  val funct    = UInt(7.W)
  val funct3   = UInt(3.W)
  val rs2      = Bits(5.W)
  val rs1      = Bits(5.W)
  val xd       = Bool()
  val xs1      = Bool()
  val xs2      = Bool()
  val rd       = Bits(5.W)
  val opcode   = UInt(7.W)
  val rs1Data  = UInt(xLen.W)
  val rs2Data  = UInt(xLen.W)
}
