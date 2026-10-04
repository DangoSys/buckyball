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
 * There is no TLB, SFENCE, access splitting, PMP unit, or implicit PTE update.
 */
@instantiable
class Walker(p: Params = Params()) extends Module {

  @public
  val io = IO(new Bundle {
    val config = Input(new Config)
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
    when(r.privilege === 3.U || io.config.mode === 0.U) {
      finish(r.vaddr(p.addressBits - 1, 0), false.B, r.vaddr(63, p.addressBits).orR, 0.U)
    }.elsewhen(r.vaddr(63, 39) =/= Fill(25, r.vaddr(38))) {
      finish(0.U, true.B, false.B, 2.U)
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
    val pte                 = io.result.bits.data
    val ptePpn              = pte(53, 10)
    val invalid             = !pte(0) || (pte(2) && !pte(1)) || pte(63, 54).orR
    val leaf                = pte(1) || pte(3)
    val supervisor          = request.privilege === 1.U
    val userPermission      = Mux(supervisor, !pte(4) || (request.sum && !request.execute), pte(4))
    val operationPermission = Mux(request.execute, pte(3), Mux(request.write, pte(2), pte(1) || (request.mxr && pte(3))))
    val misaligned          = MuxLookup(level, false.B)(Seq(
      2.U -> ptePpn(17, 0).orR,
      1.U -> ptePpn(8, 0).orR
    ))
    val leafAddress         = MuxLookup(level, Cat(ptePpn, request.vaddr(11, 0)))(Seq(
      2.U -> Cat(ptePpn(43, 18), request.vaddr(29, 0)),
      1.U -> Cat(ptePpn(43, 9), request.vaddr(20, 0))
    ))
    val leafFault           = !userPermission || !operationPermission || misaligned ||
      !pte(6) || (request.write && !pte(7))
    val leafOverflow        = leafAddress(55, p.addressBits).orR
    when(io.result.bits.error) {
      finish(0.U, false.B, true.B, level)
    }.elsewhen(invalid) {
      finish(0.U, true.B, false.B, level)
    }.elsewhen(leaf) {
      finish(leafAddress(p.addressBits - 1, 0), leafFault, !leafFault && leafOverflow, level)
    }.elsewhen(level === 0.U || pte(7, 6).orR || pte(4)) {
      finish(0.U, true.B, false.B, level)
    }.otherwise {
      ppn   := ptePpn
      level := level - 1.U
      state := issue
    }
  }

  io.resp.valid            := state === respond
  io.resp.bits             := response
  when(io.resp.fire)(state := idle)
}
