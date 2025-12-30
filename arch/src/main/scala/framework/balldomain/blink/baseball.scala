package framework.balldomain.blink

import chisel3._
import chisel3.util._

// Base trait for Ball devices
trait BallRegist {
  def Blink:  BlinkIO
  def ballId: UInt
}
