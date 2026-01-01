package framework.balldomain.blink

import chisel3._
import chisel3.util._

trait BallRegist {
  def Blink:  BlinkIO
  def ballId: UInt
}
