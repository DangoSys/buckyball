//===----------------------------------------------------------------------===//
// VVV Bond:
// Input: Vec, Vec
// Output: Vec
//===----------------------------------------------------------------------===//

package prototype.vector.bond

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config._
import prototype.vector.thread.{BaseThread, ThreadBondKey, ThreadKey}

class VVV(implicit p: Parameters) extends Bundle {
  val lane        = p(ThreadKey).get.lane
  val bondParam   = p(ThreadBondKey).get
  val inputWidth  = bondParam.inputWidth
  val outputWidth = bondParam.outputWidth

  // Input interface (Flipped Decoupled)
  val in = Flipped(Decoupled(new Bundle {
    val in1 = Vec(lane, UInt(inputWidth.W))
    val in2 = Vec(lane, UInt(inputWidth.W))
  }))

  // Decoupled output interface
  val out = Decoupled(new Bundle {
    val out = Vec(lane, UInt(outputWidth.W))
  })

}

trait CanHaveVVVBond { this: BaseThread =>

  val vvvBond = params(ThreadBondKey).filter(_.bondType == "vvv").map { bondParam =>
    // println(s"[VVVBond] Creating BondType: ${bondParam.bondType}")

    IO(new VVV()(params))
  }

  def getVVVBond = vvvBond
}
