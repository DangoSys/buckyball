// See LICENSE.SiFive for the Rocket interface definitions.
package freechips.rocketchip.tile

import chisel3._
import org.chipsalliance.cde.config.Parameters

class NMI(val w: Int) extends Bundle {
  val rnmi                  = Bool()
  val rnmi_interrupt_vector = UInt(w.W)
  val rnmi_exception_vector = UInt(w.W)
}

class TileInterrupts(implicit p: Parameters) extends CoreBundle()(p) {
  val debug = Bool()
  val mtip  = Bool()
  val msip  = Bool()
  val meip  = Bool()
  val seip  = Option.when(usingSupervisor)(Bool())
  val lip   = Vec(coreParams.nLocalInterrupts, Bool())
  val nmi   = Option.when(usingNMI)(new NMI(resetVectorLen))
}
