package framework.core.bbtile

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile.{CoreBundle, CustomCSRIO}
import freechips.rocketchip.rocket.HellaCacheIO

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

/** RoCC core IO — used inside Rocket core. */
class RoCCCoreIOBB(val nRoCCCSRs: Int = 0)(implicit p: Parameters) extends CoreBundle()(p) {
  val cmd       = Flipped(Decoupled(new RoCCCommandBB))
  val resp      = Decoupled(new RoCCResponseBB)
  val mem       = new HellaCacheIO
  val busy      = Output(Bool())
  val interrupt = Output(Bool())
  val exception = Input(Bool())
  val csrs      = Flipped(Vec(nRoCCCSRs, new CustomCSRIO))
}
