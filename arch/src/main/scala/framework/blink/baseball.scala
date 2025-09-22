package framework.blink

import chisel3._
import chisel3.util._
import org.chipsalliance.cde.config.Parameters
import examples.BuckyBallConfigs.CustomBuckyBallConfig
import examples.toy.balldomain.rs.{BallRsIssue, BallRsComplete}
import framework.builtin.memdomain.mem.{SramReadIO, SramWriteIO}

// Ball设备的基础trait
trait BallDevice {
  def Blink: Blink
  def ballId: UInt
}
