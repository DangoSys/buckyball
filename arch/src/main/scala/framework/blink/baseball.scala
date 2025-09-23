package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters

// Ball设备的基础trait
trait BallRegist {
  def Blink: Blink
  def ballId: UInt
}
