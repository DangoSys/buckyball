package memcore.memory.coherence

import chisel3._
import chisel3.util._

import memcore.bus.chi.ChiParams
import memcore.memory.cache.{CacheAccess, CacheAtomic, CacheResult}

/** Virtual-memory request used by the ROOT-native DTLB/ITLB path. */
class RootTranslationRequest extends Bundle {
  val vaddr   = UInt(64.W)
  val write   = Bool()
  val execute = Bool()
  val user    = Bool()
}

class RootTranslationResponse(p: ChiParams) extends Bundle {
  val paddr       = UInt(p.addressBits.W)
  val pageFault   = Bool()
  val accessFault = Bool()
  val level       = UInt(2.W)
}

class RootPageTableConfig extends Bundle {
  val mode    = UInt(4.W)
  val rootPpn = UInt(44.W)
}

/**
 * ROOT-native Sv39 page-table walker.
 *
 * A translation holds one ROOT cache request at a time.  The walker deliberately
 * faults A/D-bit updates instead of performing implicit writes, so the ownership
 * and ordering of those writes can be added explicitly with the DTLB policy.
 */
class RootPageTableWalker(p: ChiParams = ChiParams()) extends Module {

  val io = IO(new Bundle {
    val config = Input(new RootPageTableConfig)
    val req    = Flipped(Decoupled(new RootTranslationRequest))
    val resp   = Decoupled(new RootTranslationResponse(p))
    val access = Decoupled(new CacheAccess(p))
    val result = Flipped(Decoupled(new CacheResult))
  })

  val idle :: issue :: waitResult :: respond :: Nil = Enum(4)
  val state                                         = RegInit(idle)
  val request                                       = Reg(new RootTranslationRequest)
  val ppn                                           = Reg(UInt(44.W))
  val level                                         = RegInit(0.U(2.W))
  val response                                      = Reg(new RootTranslationResponse(p))

  private def vpn(vaddr: UInt, walkLevel: UInt): UInt =
    MuxLookup(walkLevel, vaddr(20, 12))(Seq(
      2.U -> vaddr(38, 30),
      1.U -> vaddr(29, 21),
      0.U -> vaddr(20, 12)
    ))

  private def finish(
    paddr:       UInt,
    pageFault:   Bool,
    accessFault: Bool,
    walkLevel:   UInt
  ): Unit = {
    response.paddr       := paddr
    response.pageFault   := pageFault
    response.accessFault := accessFault
    response.level       := walkLevel
    state                := respond
  }

  io.req.ready := state === idle
  when(io.req.fire) {
    request := io.req.bits
    when(io.config.mode === 0.U) {
      finish(io.req.bits.vaddr(p.addressBits - 1, 0), false.B, false.B, 0.U)
    }.elsewhen(io.config.mode === 8.U) {
      ppn   := io.config.rootPpn
      level := 2.U
      state := issue
    }.otherwise {
      finish(0.U, true.B, false.B, 0.U)
    }
  }

  io.access.valid       := state === issue
  io.access.bits.addr   := (ppn << 12) | (vpn(request.vaddr, level) << 3)
  io.access.bits.write  := false.B
  io.access.bits.data   := 0.U
  io.access.bits.mask   := 0.U
  io.access.bits.atomic := CacheAtomic.None.U
  when(io.access.fire) {
    state := waitResult
  }

  io.result.ready := state === waitResult
  when(io.result.fire) {
    val pte                 = io.result.bits.data
    val valid               = pte(0) && !(pte(2) && !pte(1))
    val leaf                = pte(1) || pte(3)
    val ptePpn              = pte(53, 10)
    val leafOffsetBits      = MuxLookup(level, 12.U)(Seq(
      2.U -> 30.U,
      1.U -> 21.U,
      0.U -> 12.U
    ))
    val offsetMask          = (1.U(64.W) << leafOffsetBits) - 1.U
    val superpageBits       = MuxLookup(level, 0.U)(Seq(
      2.U -> 18.U,
      1.U -> 9.U,
      0.U -> 0.U
    ))
    val superpageMisaligned = (ptePpn & ((1.U(44.W) << superpageBits) - 1.U)).orR
    val readable            = pte(1)
    val writable            = pte(2)
    val executable          = pte(3)
    val userAllowed         = !request.user || pte(4)
    val permission          = Mux(request.execute, executable, Mux(request.write, writable, readable)) && userAllowed
    val accessed            = pte(6)
    val dirty               = pte(7)

    when(io.result.bits.error) {
      finish(0.U, false.B, true.B, level)
    }.elsewhen(!valid) {
      finish(0.U, true.B, false.B, level)
    }.elsewhen(leaf) {
      val paddr = ((ptePpn << 12) & ~offsetMask) | (request.vaddr & offsetMask)
      finish(
        paddr(p.addressBits - 1, 0),
        !permission || !accessed || (request.write && !dirty) || superpageMisaligned,
        false.B,
        level
      )
    }.elsewhen(level === 0.U) {
      finish(0.U, true.B, false.B, level)
    }.otherwise {
      ppn   := ptePpn
      level := level - 1.U
      state := issue
    }
  }

  io.resp.valid := state === respond
  io.resp.bits  := response
  when(io.resp.fire) {
    state := idle
  }
}
