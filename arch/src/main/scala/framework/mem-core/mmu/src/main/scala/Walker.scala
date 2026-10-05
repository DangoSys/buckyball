package memcore.memory.mmu

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import memcore.bus.chi.Params
import memcore.bus.chi.rnf.{CacheAccess, CacheAtomic, CacheResult}

/** One address translation; the caller resolves effective privilege (including MPRV). */
class Request extends Bundle {
  val vaddr     = UInt(64.W)
  val write     = Bool()
  val execute   = Bool()
  val privilege = UInt(2.W) // U=0, S=1, M=3.
  val sum       = Bool()
  val mxr       = Bool()
}

class Response(p: Params) extends Bundle {
  val paddr       = UInt(p.addressBits.W)
  val pageFault   = Bool()
  val accessFault = Bool()
  val level       = UInt(2.W)
}

class Config extends Bundle {
  val mode    = UInt(4.W)
  val rootPpn = UInt(44.W)
}

/**
 * Standalone Sv39/Bare, little-endian walker with Svade-style A/D faults.
 * The CPU owns satp WARL filtering; this boundary accepts only MODE 0 or 8.
 * There is no access splitting, PMP unit, or implicit PTE update. With `entries` > 0 it keeps that
 * many fault-free leaf PTEs (a fully associative TLB, round-robin replacement) until `flush`
 * (SFENCE.VMA); entries are tagged by satp, and every hit re-checks the request's permissions
 * against the cached PTE bits. `lookup` answers the same translation combinationally, without a
 * walk, so a caller can take a fast path on a hit.
 */
@instantiable
class Walker(p: Params = Params(), entries: Int = 0) extends Module {

  @public
  val io = IO(new Bundle {
    val config = Input(new Config)
    val flush  = Input(Bool())

    val lookup = new Bundle {
      val config = Input(new Config)
      val req    = Input(new Request)
      val hit    = Output(Bool())
      val paddr  = Output(UInt(p.addressBits.W))
    }

    val req    = Flipped(Decoupled(new Request))
    val resp   = Decoupled(new Response(p))
    val access = Decoupled(new CacheAccess(p))
    val result = Flipped(Decoupled(new CacheResult))
  })

  val idle :: issue :: waitResult :: respond :: Nil = Enum(4)
  val state                                         = RegInit(idle)
  val request                                       = Reg(new Request)
  val ppn                                           = Reg(UInt(44.W))
  val level                                         = RegInit(0.U(2.W))
  val response                                      = Reg(new Response(p))

  private def vpn(address: UInt, walkLevel: UInt): UInt =
    MuxLookup(walkLevel, address(20, 12))(Seq(2.U -> address(38, 30), 1.U -> address(29, 21)))

  /** Leaf address, permission/A/D fault and physical-width overflow for one request. */
  private def leafOf(pte: UInt, walkLevel: UInt, r: Request): (UInt, Bool, Bool) = {
    val ptePpn              = pte(53, 10)
    val supervisor          = r.privilege === 1.U
    val userPermission      = Mux(supervisor, !pte(4) || (r.sum && !r.execute), pte(4))
    val operationPermission = Mux(r.execute, pte(3), Mux(r.write, pte(2), pte(1) || (r.mxr && pte(3))))
    val misaligned          = MuxLookup(walkLevel, false.B)(Seq(
      2.U -> ptePpn(17, 0).orR,
      1.U -> ptePpn(8, 0).orR
    ))
    val address             = MuxLookup(walkLevel, Cat(ptePpn, r.vaddr(11, 0)))(Seq(
      2.U -> Cat(ptePpn(43, 18), r.vaddr(29, 0)),
      1.U -> Cat(ptePpn(43, 9), r.vaddr(20, 0))
    ))
    val fault               = !userPermission || !operationPermission || misaligned || !pte(6) || (r.write && !pte(7))
    (address, fault, address(55, p.addressBits).orR)
  }

  // Leaf TLB, each entry tagged by satp and the VPN bits its level translates.
  class Entry extends Bundle {
    val mode  = UInt(4.W)
    val root  = UInt(44.W)
    val vpn   = UInt(27.W)
    val level = UInt(2.W)
    val pte   = UInt(64.W)
  }

  val tlbValid  = RegInit(VecInit(Seq.fill(entries max 1)(false.B)))
  val tlb       = Reg(Vec(entries max 1, new Entry))
  val tlbVictim = RegInit(0.U(log2Ceil(entries max 2).W))

  private def tlbMatches(vaddr: UInt, config: Config): Seq[Bool] = tlb.zip(tlbValid).map { case (e, valid) =>
    val vpn = vaddr(38, 12)
    (entries > 0).B && valid && e.mode === config.mode && e.root === config.rootPpn &&
    MuxLookup(e.level, vpn === e.vpn)(Seq(
      2.U -> (vpn(26, 18) === e.vpn(26, 18)),
      1.U -> (vpn(26, 9) === e.vpn(26, 9))
    ))
  }

  /** (hit, address, page fault, access fault, level) for a request served without a walk. */
  private def translate(r: Request, config: Config): (Bool, UInt, Bool, Bool, UInt) = {
    val identity                   = r.privilege === 3.U || config.mode === 0.U
    val canonical                  = r.vaddr(63, 39) === Fill(25, r.vaddr(38))
    val matches                    = tlbMatches(r.vaddr, config)
    val entry                      = PriorityMux(matches, tlb)
    val (address, fault, overflow) = leafOf(entry.pte, entry.level, r)
    val hit                        = identity || !canonical || matches.reduce(_ || _)
    val paddr                      = Mux(identity, r.vaddr(p.addressBits - 1, 0), address(p.addressBits - 1, 0))
    val pageFault                  = !identity && (!canonical || fault)
    val access                     = Mux(identity, r.vaddr(63, p.addressBits).orR, canonical && !fault && overflow)
    (hit, paddr, pageFault, access, Mux(identity, 0.U, Mux(canonical, entry.level, 2.U)))
  }

  locally {
    val (hit, paddr, pageFault, accessFault, _) = translate(io.lookup.req, io.lookup.config)
    io.lookup.hit   := hit && !pageFault && !accessFault
    io.lookup.paddr := paddr
  }

  private def finish(
    address:     UInt,
    pageFault:   Bool,
    accessFault: Bool,
    walkLevel:   UInt
  ): Unit = {
    response.paddr       := Mux(pageFault || accessFault, 0.U, address)
    response.pageFault   := pageFault
    response.accessFault := accessFault
    response.level       := walkLevel
    state                := respond
  }

  io.req.ready := state === idle
  when(io.req.fire) {
    val r = io.req.bits
    assert(io.config.mode === 0.U || io.config.mode === 8.U, "Walker requires Bare or Sv39 mode")
    assert(
      r.privilege === 0.U || r.privilege === 1.U || r.privilege === 3.U,
      "Walker effective privilege must be U, S or M"
    )
    assert(!(r.write && r.execute), "Walker access cannot both write and execute")
    request := r
    val (hit, paddr, pageFault, accessFault, walkLevel) = translate(r, io.config)
    when(hit) {
      finish(paddr, pageFault, accessFault, walkLevel)
    }.otherwise {
      ppn   := io.config.rootPpn
      level := 2.U
      state := issue
    }
  }

  // Keep all Sv39 physical bits until the implemented address width is checked.
  // A PTE index occupies exactly the low 12 bits of its 4-KiB table page.
  val pteAddress         = Cat(ppn, vpn(request.vaddr, level), 0.U(3.W))
  val pteAddressOverflow = pteAddress(55, p.addressBits).orR
  io.access.valid           := state === issue && !pteAddressOverflow
  io.access.bits.addr       := pteAddress(p.addressBits - 1, 0)
  io.access.bits.write      := false.B
  io.access.bits.data       := 0.U
  io.access.bits.mask       := 0.U
  io.access.bits.atomic     := CacheAtomic.None.U
  io.access.bits.atomicWord := false.B
  when(state === issue && pteAddressOverflow) {
    finish(0.U, false.B, true.B, level)
  }.elsewhen(io.access.fire) {
    state := waitResult
  }

  io.result.ready := state === waitResult
  when(io.result.fire) {
    val pte                                    = io.result.bits.data
    val ptePpn                                 = pte(53, 10)
    val invalid                                = !pte(0) || (pte(2) && !pte(1)) || pte(63, 54).orR
    val leaf                                   = pte(1) || pte(3)
    val (leafAddress, leafFault, leafOverflow) = leafOf(pte, level, request)
    when(io.result.bits.error) {
      finish(0.U, false.B, true.B, level)
    }.elsewhen(invalid) {
      finish(0.U, true.B, false.B, level)
    }.elsewhen(leaf) {
      finish(leafAddress(p.addressBits - 1, 0), leafFault, !leafFault && leafOverflow, level)
      if (entries > 0) when(!leafFault && !leafOverflow) {
        tlbValid(tlbVictim)  := true.B
        tlb(tlbVictim).mode  := io.config.mode
        tlb(tlbVictim).root  := io.config.rootPpn
        tlb(tlbVictim).vpn   := request.vaddr(38, 12)
        tlb(tlbVictim).level := level
        tlb(tlbVictim).pte   := pte
        tlbVictim            := Mux(tlbVictim === (entries - 1).U, 0.U, tlbVictim + 1.U)
      }
    }.elsewhen(level === 0.U || pte(7, 6).orR || pte(4)) {
      finish(0.U, true.B, false.B, level)
    }.otherwise {
      ppn   := ptePpn
      level := level - 1.U
      state := issue
    }
  }

  // SFENCE.VMA wins over a same-cycle fill.
  when(io.flush)(tlbValid.foreach(_ := false.B))

  io.resp.valid            := state === respond
  io.resp.bits             := response
  when(io.resp.fire)(state := idle)
}
