package framework.balldomain.blink

import chisel3._
import chisel3.util._

class BallStatus extends Bundle {
  val idle    = Output(Bool())
  val running = Output(Bool())
}
