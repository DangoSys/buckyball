package framework.memdomain.frontend.mem

import chisel3._
import chisel3.util._
import framework.top.GlobalConfig
import framework.memdomain.frontend.mem.dma.DmaStatus

class Footprint(b: GlobalConfig) extends Bundle {
  val valid        = Bool()
  val rob_id       = UInt(log2Up(b.frontend.rob_entries).W)
  val is_sub       = Bool()
  val sub_rob_id   = UInt(log2Up(b.frontend.sub_rob_depth * 4).W)
  val baseVA       = UInt(64.W)
  val rows         = UInt(64.W)
  val columns      = UInt(64.W)
  val spanBytes    = UInt(64.W)
  val columnStride = UInt(64.W)
  val rowStride    = UInt(64.W)
  val write        = Bool()
  val fault        = new DmaStatus
}
