//===- BaseThread.scala - Level 1: Thread ---===//
package examples.balls.vector.thread

import chisel3.experimental.hierarchy.{instantiable, public}

import chisel3._
import examples.balls.vector.configs.VectorBallParam

//===----------------------------------------------------------------------===//
// BaseThread base class
//===----------------------------------------------------------------------===//
@instantiable
class BaseThread(val opType: String, ballParam: VectorBallParam) extends Module {
  val config = ballParam

  @public
  val io   = IO(new Bundle {})
  val lane = config.lane
}
