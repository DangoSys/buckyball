//===- BaseThread.scala - Level 1: Thread ---===//
package framework.balldomain.prototype.vector.thread

import chisel3._
import framework.balldomain.prototype.vector.configs.VectorBallParam

//===----------------------------------------------------------------------===//
// BaseThread base class
//===----------------------------------------------------------------------===//
class BaseThread(val config: VectorBallParam, val opType: String) extends Module {
  val io   = IO(new Bundle {})
  val lane = config.lane
}
