package framework.memdomain.frontend.outside_channel.tlb

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
  val pf       = Bool()
  val ae_final = Bool()
}

/**
 * TLB entry containing VPN tag and entry data.
 *
 * Supports superpages. The `level` field stores the PTW response level,
 * which indicates at which page table walk step the PTE was found.
 * For Sv39 with pgLevels=4 (matching Rocket PTW):
 *   level=1 → 1GB gigapage (only VPN[2] in PPN, VPN[1:0] from VA)
 *   level=2 → 2MB megapage (VPN[2:1] in PPN, VPN[0] from VA)
 *   level=3 → 4KB page    (full PPN from PTE)
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
    // Walk from highest VPN chunk (j=1) to lowest (j=pgLevels-1),
    // same ordering as Rocket TLB's ppn() method.
    val matches = (1 until pgLevels).map { j =>
      // Rocket convention: j=1 is the highest VPN/PPN chunk below the top,
      // j=pgLevels-1 is the lowest.
      // supervisorVPNBits = pgLevels * pgLevelBits (may exceed actual vpnBits)
      val supervisorVPNBits = pgLevels * pgLevelBits
      val hi                = supervisorVPNBits - j * pgLevelBits - 1
      val lo                = supervisorVPNBits - (j + 1) * pgLevelBits
      // Clamp to actual vpnBits
      val clampedHi         = math.min(hi, vpnBits - 1)
      val clampedLo         = math.max(lo, 0)
      if (clampedHi < clampedLo) {
        true.B // Chunk doesn't exist in VPN, always matches
      } else {
        val ignore = level < j.U
        ignore || (vpn(clampedHi, clampedLo) === tag_vpn(clampedHi, clampedLo))
      }
    }
    valid && matches.reduce(_ && _)
  }

  /**
   * Generate the correct PPN for superpage translation.
   * Follows Rocket TLB's ppn() method exactly.
   */
  def ppn(vpn: UInt): UInt = {
    val supervisorVPNBits = pgLevels * pgLevelBits
    // Start with the highest PPN chunk (above all VPN levels)
    var res               = data.ppn >> (pgLevelBits * (pgLevels - 1))
    for (j <- 1 until pgLevels) {
      val hi    = supervisorVPNBits - j * pgLevelBits - 1
      val lo    = supervisorVPNBits - (j + 1) * pgLevelBits
      // Clamp to actual ppnBits and vpnBits
      val ppnHi = math.min(hi, ppnBits - 1)
      val ppnLo = math.max(lo, 0)
      if (ppnHi >= ppnLo) {
        val ignore = level < j.U
        val vpnHi  = math.min(hi, vpnBits - 1)
        val vpnLo  = math.max(lo, 0)
        val chunk  = Mux(ignore, vpn(vpnHi, vpnLo), data.ppn(ppnHi, ppnLo))
        res = Cat(res, chunk)
      }
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
