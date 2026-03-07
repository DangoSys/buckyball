package framework.memdomain.frontend.outside_channel.tlb

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

/** TLB implementation with fully-associative structure and LRU replacement */
@instantiable
class TLB(val b: GlobalConfig, val lgMaxSize: Int) extends Module {
  val entries   = b.memDomain.tlb_size
  val vaddrBits = b.core.vaddrBits
  val paddrBits = b.core.paddrBits
  val pgIdxBits = b.core.pgIdxBits
  val vpnBits   = vaddrBits - pgIdxBits
  val ppnBits   = paddrBits - pgIdxBits
  val pgLevels  = if (b.core.xLen == 32) 2 else 4 // Match Rocket PTW convention

  @public
  val io = IO(new Bundle {
    val req    = Flipped(Decoupled(new BBTLBReq(lgMaxSize, vaddrBits, b.core.xLen)))
    val resp   = Decoupled(new BBTLBResp(lgMaxSize, paddrBits, vaddrBits))
    val ptw    = new BBTLBPTWIO(b)
    val sfence = Flipped(Valid(Bool())) // Simplified flush signal
    val kill   = Input(Bool())
  })

  // TLB entries storage
  val tlbEntries = Reg(Vec(entries, new TLBEntry(vaddrBits, pgIdxBits, paddrBits, pgLevels = pgLevels)))
  val lru        = Reg(Vec(entries, UInt(log2Ceil(entries).W))) // Simple LRU counter

  // State machine
  val s_ready :: s_request :: s_wait :: Nil = Enum(3)
  val state                                 = RegInit(s_ready)
  val refill_vpn                            = Reg(UInt(vpnBits.W))
  val refill_idx                            = Reg(UInt(log2Ceil(entries).W))

  // Initialize LRU
  when(reset.asBool) {
    lru.foreach(_ := 0.U)
    tlbEntries.foreach(_.invalidate())
  }

  val vpn   = io.req.bits.vaddr(vaddrBits - 1, pgIdxBits)
  val pgIdx = io.req.bits.vaddr(pgIdxBits - 1, 0)

  // TLB lookup
  val hits   = tlbEntries.map(_.hit(vpn))
  val hitVec = VecInit(hits)
  val hitIdx = PriorityEncoder(hits)
  val tlbHit = hits.reduce(_ || _)

  // Update LRU on hit
  when(io.req.fire && tlbHit) {
    lru(hitIdx) := (entries - 1).U
    for (i <- 0 until entries) {
      when(i.U =/= hitIdx && lru(i.U) > lru(hitIdx)) {
        lru(i.U) := lru(i.U) - 1.U
      }
    }
  }

  // Find LRU entry for replacement
  val lruIdx = PriorityEncoder(lru.map(_ === 0.U))

  // VM enable check: mode != 0 means virtual memory is on (Sv39/Sv48/etc.)
  val vm_enabled = io.ptw.ptbr.mode =/= 0.U

  val tlbMiss = vm_enabled && !io.req.bits.passthrough && !tlbHit

  // State machine
  io.req.ready := state === s_ready

  when(io.req.fire && tlbMiss && state === s_ready) {
    state      := s_request
    refill_vpn := vpn
    refill_idx := lruIdx
  }

  when(state === s_request) {
    when(io.kill) {
      state := s_ready
    }.elsewhen(io.ptw.req.ready) {
      state := s_wait
    }
  }

  when(state === s_wait && io.ptw.resp.valid) {
    state := s_ready
    // Refill TLB entry
    val pte       = io.ptw.resp.bits.pte
    val entryData = Wire(new TLBEntryData(paddrBits, pgIdxBits))
    entryData.ppn       := pte.ppn(ppnBits - 1, 0)
    entryData.u         := pte.u
    entryData.g         := pte.g
    entryData.sr        := pte.sr()
    entryData.sw        := pte.sw()
    entryData.sx        := pte.sx()
    entryData.cacheable := true.B // Simplified
    entryData.pf        := io.ptw.resp.bits.pf
    entryData.ae_final  := io.ptw.resp.bits.ae_final

    tlbEntries(refill_idx).insert(refill_vpn, io.ptw.resp.bits.level, entryData)
    // Update LRU
    lru(refill_idx) := (entries - 1).U
    for (i <- 0 until entries) {
      when(i.U =/= refill_idx && lru(i.U) > lru(refill_idx)) {
        lru(i.U) := lru(i.U) - 1.U
      }
    }
  }

  // PTW request
  io.ptw.req.valid              := state === s_request
  io.ptw.req.bits.valid         := !io.kill
  io.ptw.req.bits.bits.addr     := refill_vpn
  io.ptw.req.bits.bits.vstage1  := false.B
  io.ptw.req.bits.bits.stage2   := false.B
  io.ptw.req.bits.bits.need_gpa := false.B

  // TLB flush on sfence
  when(io.sfence.valid) {
    tlbEntries.foreach(_.invalidate())
    state := s_ready
  }

  // Response generation — use superpage-aware ppn from the hit entry
  val hitEntryData = Mux(
    tlbHit,
    tlbEntries(hitIdx).data,
    WireDefault(new TLBEntryData(paddrBits, pgIdxBits), 0.U.asTypeOf(new TLBEntryData(paddrBits, pgIdxBits)))
  )

  // For superpage entries, TLBEntry.ppn() replaces lower PPN bits with VPN bits
  val hitPPN = Mux(
    tlbHit,
    tlbEntries(hitIdx).ppn(vpn),
    0.U(ppnBits.W)
  )

  val paddr = Mux(
    io.req.bits.passthrough || !vm_enabled,
    io.req.bits.vaddr,
    Cat(hitPPN, pgIdx)
  )

  io.resp.valid             := io.req.valid && state === s_ready
  io.resp.bits.miss         := tlbMiss || (state === s_wait)
  io.resp.bits.paddr        := paddr(paddrBits - 1, 0)
  io.resp.bits.gpa          := 0.U
  io.resp.bits.gpa_is_pte   := false.B
  io.resp.bits.pf.ld        := hitEntryData.pf
  io.resp.bits.pf.st        := hitEntryData.pf
  io.resp.bits.pf.inst      := hitEntryData.pf
  io.resp.bits.gf.ld        := false.B
  io.resp.bits.gf.st        := false.B
  io.resp.bits.gf.inst      := false.B
  io.resp.bits.ae.ld        := hitEntryData.ae_final
  io.resp.bits.ae.st        := hitEntryData.ae_final
  io.resp.bits.ae.inst      := hitEntryData.ae_final
  io.resp.bits.ma.ld        := false.B
  io.resp.bits.ma.st        := false.B
  io.resp.bits.ma.inst      := false.B
  io.resp.bits.cacheable    := hitEntryData.cacheable
  io.resp.bits.must_alloc   := false.B
  io.resp.bits.prefetchable := hitEntryData.cacheable
  io.resp.bits.size         := io.req.bits.size
  io.resp.bits.cmd          := io.req.bits.cmd
}
