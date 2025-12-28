//===----------------------------------------------------------------------===//
// VVV Bond:
// Input: Vec, Vec
// Output: Vec
//===----------------------------------------------------------------------===//

package framework.balldomain.prototype.vector.bond

import chisel3._
import chisel3.util._
import framework.balldomain.prototype.vector.configs.VectorBallParam
import chisel3.experimental.hierarchy.{instantiable, public}

@instantiable
class VVV(val config: VectorBallParam, val inputWidth: Int, val outputWidth: Int) extends Bundle {
  val lane = config.lane

  @public
  val in = Flipped(Decoupled(new Bundle {
    val in1 = Vec(lane, UInt(inputWidth.W))
    val in2 = Vec(lane, UInt(inputWidth.W))
  }))

  @public
  val out = Decoupled(new Bundle {
    val out = Vec(lane, UInt(outputWidth.W))
  })

}
