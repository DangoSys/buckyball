package examples.toy

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters

object FenceCSR {
  def apply(): UInt = RegInit(0.U(64.W))
}
