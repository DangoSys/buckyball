package framework.memdomain.backend.banks.btrace

import chisel3._
import framework.top.GlobalConfig

class BTraceRecord extends Bundle {
  val instId  = UInt(64.W)
  val hartId  = UInt(64.W)
  val w0Vbank = UInt(32.W)
  val w0Hash  = UInt(32.W)
}

class PhysicalBankHash(val b: GlobalConfig) extends Bundle {
  val valid      = Bool()
  val hartId     = UInt(b.tile.xLen.W)
  val vbankId    = UInt(b.memDomain.vbankIdWidth.W)
  val pbankId    = UInt(32.W)
  val groupId    = UInt(32.W)
  val statusHash = UInt(32.W)
}
