package framework.balldomain.blink

import chisel3._
import chisel3.util._

// all balls must have a blink interface
trait HasBlink {
  def blink: BlinkIO
}

// can be customized by user to add additional status signals
trait HasBallStatus {
  def status: BallStatus
}
