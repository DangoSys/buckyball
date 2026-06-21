package framework.memdomain.frontend.mem.tlb

import chisel3._
import chisel3.util._

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
  val pf     = Bool()
  val gf     = Bool()
  val ae_ptw = Bool()
  val ae     = Bool()
}

/**
 * TLB entry containing VPN tag and entry data.
 *
 * Supports superpages. The `level` field stores the PTW response level,
 * which indicates at which page table walk step the PTE was found.
 * For Sv39 with pgLevels=3 (matching Rocket PTW):
 *   level=0 → 1GB gigapage (PPN[1:0] from VPN[1:0])
 *   level=1 → 2MB megapage (PPN[0] from VPN[0])
 *   level=2 → 4KB page    (full PPN from PTE)
 *
 * The superpage PPN generation follows Rocket TLB's logic:
 *   for j in 1 until pgLevels:
 *     if level < j: use VPN chunk (superpage covers this level)
 *     else:         use PPN chunk from PTE
 */
class TLBEntry(
  val vaddrBits:   Int,
  val pgIdxBits:   Int,
  val paddrBits:   Int,
  val pgLevelBits: Int = 9,
  val pgLevels:    Int = 4)
    extends Bundle {
  val vpnBits = vaddrBits - pgIdxBits
  val ppnBits = paddrBits - pgIdxBits

  val tag_vpn = UInt(vpnBits.W)
  val valid   = Bool()
  val level   = UInt(log2Ceil(pgLevels).W)
  val data    = new TLBEntryData(paddrBits, pgIdxBits)

  /**
   * Superpage-aware hit: only compare VPN chunks that are above the
   * superpage boundary. Follows Rocket's convention where `level < j`
   * means the chunk at position j is covered by the superpage.
   */
  def hit(vpn: UInt): Bool = {
    val matches = (0 until pgLevels).map { j =>
      val base   = (pgLevels - 1 - j) * pgLevelBits
      val hi     = math.min(base + pgLevelBits - 1, vpnBits - 1)
      val lo     = math.max(base, 0)
      val ignore = level < j.U
      if (hi < lo) true.B else ignore || ((tag_vpn ^ vpn)(hi, lo) === 0.U)
    }
    valid && matches.reduce(_ && _)
  }

  /**
   * Generate the correct PPN for superpage translation.
   * Follows Rocket TLB's ppn() method exactly.
   */
  def ppn(vpn: UInt): UInt = {
    val supervisorVPNBits = pgLevels * pgLevelBits
    var res               = data.ppn >> (pgLevelBits * (pgLevels - 1))
    for (j <- 1 until pgLevels) {
      val hi     = supervisorVPNBits - j * pgLevelBits - 1
      val lo     = supervisorVPNBits - (j + 1) * pgLevelBits
      val ignore = level < j.U
      res = Cat(res, (Mux(ignore, vpn, 0.U) | data.ppn)(hi, lo))
    }
    res
  }

  def insert(vpn: UInt, lvl: UInt, entryData: TLBEntryData): Unit = {
    tag_vpn := vpn
    valid   := true.B
    level   := lvl
    data    := entryData
  }

  def invalidate(): Unit =
    valid := false.B
}
