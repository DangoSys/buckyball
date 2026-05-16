package framework.system.core.scu

import chisel3._
import org.chipsalliance.cde.config.Parameters
import org.chipsalliance.diplomacy.lazymodule._
import freechips.rocketchip.tilelink._

/**
 * TLSCU - LazyModule wrapper for SCU
 *
 * Wraps the SCU Module in a LazyModule with TLAdapterNode so it can be
 * inserted into the Diplomacy graph.
 */
class TLSCU(val params: SCUParams)(implicit p: Parameters) extends LazyModule {

  val node = TLAdapterNode(
    clientFn = { p => p },
    managerFn = { p => p }
  )

  lazy val module = new Impl

  class Impl extends LazyModuleImp(this) {
    (node.in zip node.out) foreach { case ((in, edgeIn), (out, edgeOut)) =>
      val tlBundleParams = TLBundleParameters(
        addressBits = edgeOut.bundle.addressBits,
        dataBits = edgeOut.bundle.dataBits,
        sourceBits = edgeOut.bundle.sourceBits,
        sinkBits = edgeOut.bundle.sinkBits,
        sizeBits = edgeOut.bundle.sizeBits,
        echoFields = edgeOut.bundle.echoFields,
        requestFields = edgeOut.bundle.requestFields,
        responseFields = edgeOut.bundle.responseFields,
        hasBCE = edgeOut.bundle.hasBCE
      )

      val scu = Module(new SCU(params, tlBundleParams))
      scu.io.tl_in <> in
      out <> scu.io.tl_out
    }
  }

}
