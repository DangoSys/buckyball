package framework.memdomain.frontend.outside_channel.tlb

import chisel3._

/** TLB entry data containing translation and permission information */
class TLBEntryData(val paddrBits: Int, val pgIdxBits: Int) extends Bundle {
  val ppnBits = paddrBits - pgIdxBits

  val ppn       = UInt(ppnBits.W)
  val u         = Bool() // user page
  val g         = Bool() // global page
  val sr        = Bool() // supervisor read
  val sw        = Bool() // supervisor write
  val sx        = Bool() // supervisor execute
  val cacheable = Bool()

  // Page fault and access exception flags
  val pf       = Bool()
  val ae_final = Bool()
}

/** TLB entry containing VPN tag and entry data */
class TLBEntry(val vaddrBits: Int, val pgIdxBits: Int, val paddrBits: Int) extends Bundle {
  val vpnBits = vaddrBits - pgIdxBits

  val tag_vpn = UInt(vpnBits.W)
  val valid   = Bool()
  val data    = new TLBEntryData(paddrBits, pgIdxBits)

  def hit(vpn: UInt): Bool = valid && (tag_vpn === vpn)

  def insert(vpn: UInt, entryData: TLBEntryData): Unit = {
    tag_vpn := vpn
    valid   := true.B
    data    := entryData
  }

  def invalidate(): Unit =
    valid := false.B
}
