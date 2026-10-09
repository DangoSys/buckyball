// See LICENSE.Berkeley and LICENSE.SiFive for Rocket CSR semantics.
package freechips.rocketchip.rocket

import chisel3._
import org.chipsalliance.cde.config.Parameters
import freechips.rocketchip.tile._
import freechips.rocketchip.util._

trait HasRocketCoreParameters extends HasCoreParameters {
  lazy val rocketParams = tileParams.core.asInstanceOf[RocketCoreParams]
  val fastLoadWord      = rocketParams.fastLoadWord
  val fastLoadByte      = rocketParams.fastLoadByte
  val mulDivParams      = rocketParams.mulDiv.getOrElse(MulDivParams())
  require(!fastLoadByte || fastLoadWord)
  require(!rocketParams.haveFSDirty)
}

class RocketCustomCSRs(implicit p: Parameters) extends CustomCSRs with HasRocketCoreParameters {
  override def bpmCSR = rocketParams.branchPredictionModeCSR.option(CustomCSR(bpmCSRId, 1, Some(BigInt(0))))

  override def chickenCSR = {
    val dc   = tileParams.dcache.get
    val mask = BigInt(dc.clockGate.toInt | (rocketParams.clockGate.toInt << 1) |
      (rocketParams.clockGate.toInt << 2) | (1 << 3) | (dc.scratch.isEmpty.toInt << 9) |
      (tileParams.icache.get.prefetch.toInt << 17))
    Some(CustomCSR(chickenCSRId, mask, Some(mask)))
  }

  def disableICachePrefetch = getOrElse(chickenCSR, _.value(17), true.B)
  def marchid               = CustomCSR.constant(CSRs.marchid, BigInt(1))
  def mvendorid             = CustomCSR.constant(CSRs.mvendorid, BigInt(rocketParams.mvendorid))
  def mimpid                = CustomCSR.constant(CSRs.mimpid, BigInt(rocketParams.mimpid))
  override def decls        = super.decls :+ marchid :+ mvendorid :+ mimpid
}
