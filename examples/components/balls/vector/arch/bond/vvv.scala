//===----------------------------------------------------------------------===//
// VVV Bond:
// Input: Vec, Vec
// Output: Vec
//===----------------------------------------------------------------------===//

package examples.balls.vector.bond

import chisel3._
import chisel3.util._
import examples.balls.vector.configs.VectorBallParam

class VVV(val inputWidth: Int, val outputWidth: Int, val ballParam: VectorBallParam) extends Bundle {
  val config = ballParam

  val lane = config.lane

  val in = Flipped(Decoupled(new Bundle {
    val in1 = Vec(lane, UInt(inputWidth.W))
    val in2 = Vec(lane, UInt(inputWidth.W))
  }))

  val out = Decoupled(new Bundle {
    val out = Vec(lane, UInt(outputWidth.W))
  })

}
