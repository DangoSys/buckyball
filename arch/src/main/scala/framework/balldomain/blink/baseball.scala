package framework.balldomain.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters

// Base trait for Ball devices
trait BallRegist {
  def Blink:  Blink
  def ballId: UInt
}
