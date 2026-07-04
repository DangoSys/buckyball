//===- BaseThread.scala - Level 1: Thread ---===//
package examples.balls.vector.thread

import chisel3._
import examples.balls.vector.configs.VectorBallParam

//===----------------------------------------------------------------------===//
// BaseThread base class
//===----------------------------------------------------------------------===//
class BaseThread(val config: VectorBallParam, val opType: String) extends Module {
  val io   = IO(new Bundle {})
  val lane = config.lane
}
