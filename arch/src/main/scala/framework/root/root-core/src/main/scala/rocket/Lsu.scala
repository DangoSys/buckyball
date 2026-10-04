package hier.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import freechips.rocketchip.rocket._
import freechips.rocketchip.rocket.constants.MemoryOpConstants
import memcore.bus.chi.rnf.CacheAtomic
import memcore.memory.cpu.{CpuMemParams, VirtualMemoryRequest, VirtualMemoryResponse}
import memcore.memory.fetch.Context

/**
 * Same-cycle hit check for an identity-translated plain access in Rocket's s2. The owner answers
 * `hit` only after permission, region, interlock and L1 hit (including a store's write) all succeed.
 */
class LsuProbe extends Bundle {
  val valid     = Output(Bool())
  val addr      = Output(UInt(64.W))
  val size      = Output(UInt(2.W))
  val write     = Output(Bool())
  val data      = Output(UInt(64.W))
  val privilege = Output(UInt(2.W))
  val hit       = Input(Bool())
  val value     = Input(UInt(64.W))
}

/**
 * A probe hit completes in s2. Otherwise: fixed S1/S2 cancellation windows, then execute once
 * and return only on the instruction's replay.
 */
@instantiable
class Lsu(cp: CpuMemParams)(implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters
    with MemoryOpConstants {

  @public
  val io = IO(new Bundle {
    val cpu         = Flipped(new HellaCacheIO)
    val pc          = Input(UInt(vaddrBitsExtended.W))
    val pmp         = Input(Vec(nPMPs, new PMP))
    val context     = Input(new Context)
    val maintenance = Decoupled(new SFenceReq)
    val maintained  = Flipped(Decoupled(Bool()))
    val capturedPmp = Output(Vec(nPMPs, new PMP))
    val request     = Decoupled(new VirtualMemoryRequest(cp))
    val response    = Flipped(Decoupled(new VirtualMemoryResponse(cp)))
    val probe       = new LsuProbe
    val idle        = Output(Bool())
    val cancelled   = Output(Bool())
  })

  require(!usingHypervisor && cp.tagBits <= coreParams.dcacheReqTagBits)
  val idle :: s1 :: s2 :: issue :: receive :: completed :: replay1 :: replay2 :: Nil = Enum(8)
  val state                                                                          = RegInit(idle)
  val command                                                                        = Reg(new HellaCacheReq)
  val instructionPc                                                                  = Reg(UInt(vaddrBitsExtended.W))
  val data                                                                           = Reg(UInt(64.W))
  val pmps                                                                           = Reg(Vec(nPMPs, new PMP))
  val context                                                                        = Reg(new Context)
  val maintaining                                                                    = RegInit(false.B)
  io.capturedPmp := pmps
  val result  = Reg(new VirtualMemoryResponse(cp))
  val matches = io.cpu.req.bits.tag === command.tag && io.cpu.req.bits.addr === command.addr &&
    io.cpu.req.bits.cmd === command.cmd && io.cpu.req.bits.size === command.size && io.pc === instructionPc
  io.idle          := state === idle
  io.cancelled     := (state === s1 || state === replay1) && io.cpu.s1_kill ||
    (state === s2 || state === replay2) && io.cpu.s2_kill
  io.cpu.req.ready := state === idle || (state === completed && (!io.cpu.req.valid || matches))
  when(io.cpu.req.fire) {
    when(state === idle) {
      assert(!io.cpu.req.bits.dv && io.cpu.req.bits.dprv =/= 2.U, "LSU requires nonvirtual U, S or M privilege")
      if (io.cpu.req.bits.tag.getWidth > cp.tagBits) {
        assert(
          !io.cpu.req.bits.tag(io.cpu.req.bits.tag.getWidth - 1, cp.tagBits).orR,
          "LSU arbiter tag bits must be zero"
        )
      }
      command       := io.cpu.req.bits
      instructionPc := io.pc
      pmps          := io.pmp
      context       := io.context
      state         := s1
    }.otherwise {
      assert(matches, "LSU replay changed instruction identity")
      state := replay1
    }
  }
  when(state === s1) {
    when(io.cpu.s1_kill)(state := idle)
      .otherwise { data := io.cpu.s1_data.data; state := s2 }
  }

  val writes     = isWrite(command.cmd)
  val reads      = isRead(command.cmd)
  val address    = if (usingVM) command.addr.asSInt.pad(64).asUInt else command.addr.pad(64)
  val sfence     = command.cmd === M_SFENCE
  val misaligned = (address & ((1.U(64.W) << command.size) - 1.U)).orR

  // Bare translation only: M-mode effective privilege or a Bare satp makes the VA the PA.
  val identity = command.dprv === PRV.M.U || context.satp(63, 60) === 0.U
  io.probe.valid     := state === s2 && !io.cpu.s2_kill && !misaligned && identity &&
    (command.cmd === M_XRD || command.cmd === M_XWR)
  io.probe.addr      := address
  io.probe.size      := command.size
  io.probe.write     := command.cmd === M_XWR
  io.probe.data      := data
  io.probe.privilege := command.dprv
  val probed    = io.probe.valid && io.probe.hit
  val probedRaw = io.probe.value >> (address(2, 0) << 3)

  val probedLoad = MuxLookup(command.size, probedRaw)(Seq(
    0.U -> Cat(Fill(56, command.signed && probedRaw(7)), probedRaw(7, 0)),
    1.U -> Cat(Fill(48, command.signed && probedRaw(15)), probedRaw(15, 0)),
    2.U -> Cat(Fill(32, command.signed && probedRaw(31)), probedRaw(31, 0))
  ))

  when(state === s2) {
    when(io.cpu.s2_kill)(state := idle).elsewhen(probed)(state := idle).otherwise {
      when(misaligned && !sfence) {
        result             := 0.U.asTypeOf(result)
        result.tag         := command.tag(cp.tagBits - 1, 0)
        result.misaligned  := misaligned
        result.accessFault := false.B
        state              := completed
      }.otherwise(state := issue)
    }
  }

  val atom = MuxLookup(command.cmd, CacheAtomic.None.U)(Seq(
    M_XA_SWAP -> CacheAtomic.Swap.U,
    M_XA_ADD  -> CacheAtomic.Add.U,
    M_XA_XOR  -> CacheAtomic.Xor.U,
    M_XA_AND  -> CacheAtomic.And.U,
    M_XA_OR   -> CacheAtomic.Or.U,
    M_XA_MIN  -> CacheAtomic.Min.U,
    M_XA_MAX  -> CacheAtomic.Max.U,
    M_XA_MINU -> CacheAtomic.MinU.U,
    M_XA_MAXU -> CacheAtomic.MaxU.U,
    M_XLR     -> CacheAtomic.LR.U,
    M_XSC     -> CacheAtomic.SC.U
  ))

  io.maintenance.valid          := state === issue && sfence
  io.maintenance.bits           := 0.U.asTypeOf(io.maintenance.bits)
  io.maintenance.bits.addr      := command.addr
  io.maintenance.bits.asid      := data
  io.maintenance.bits.rs1       := command.size(0)
  io.maintenance.bits.rs2       := command.size(1)
  io.maintained.ready           := state === receive && maintaining
  when(io.maintenance.fire) { maintaining := true.B; state := receive }
  when(io.maintained.fire) {
    assert(io.maintained.bits, "LSU maintenance must acknowledge completion")
    result      := 0.U.asTypeOf(result)
    result.tag  := command.tag(cp.tagBits - 1, 0)
    maintaining := false.B
    state       := completed
  }
  io.request.valid              := state === issue && !sfence
  io.request.bits               := 0.U.asTypeOf(io.request.bits)
  io.request.bits.vaddr         := address
  io.request.bits.tag           := command.tag(cp.tagBits - 1, 0)
  io.request.bits.size          := command.size
  io.request.bits.write         := command.cmd === M_XWR
  io.request.bits.execute       := false.B
  io.request.bits.signed        := command.signed
  io.request.bits.data          := data
  io.request.bits.atomic        := atom
  io.request.bits.privilege     := command.dprv
  io.request.bits.sum           := context.sum
  io.request.bits.mxr           := context.mxr
  io.request.bits.satpMode      := context.satp(63, 60)
  io.request.bits.rootPpn       := context.satp(43, 0)
  when(io.request.fire) {
    assert(
      command.cmd === M_XRD || command.cmd === M_XWR || command.cmd === M_XLR ||
        command.cmd === M_XSC || isAMO(command.cmd),
      "LSU unsupported memory operation"
    )
    state := receive
  }
  io.response.ready             := state === receive && !maintaining
  when(io.response.fire) {
    assert(io.response.bits.tag === command.tag(cp.tagBits - 1, 0), "LSU virtual response tag mismatch")
    result := io.response.bits
    state  := completed
  }
  when(state === replay1)(state := Mux(io.cpu.s1_kill, completed, replay2))
  when(state === replay2)(state := Mux(io.cpu.s2_kill, completed, idle))

  val returns   = state === replay2 && !io.cpu.s2_kill
  val exception = result.misaligned || result.pageFault || result.accessFault
  io.cpu.s2_nack                    := state === s2 && !io.cpu.s2_kill && !probed
  io.cpu.s2_nack_cause_raw          := false.B
  io.cpu.s2_uncached                := false.B
  io.cpu.s2_paddr                   := command.addr(paddrBits - 1, 0)
  io.cpu.s2_gpa                     := 0.U
  io.cpu.s2_gpa_is_pte              := false.B
  io.cpu.s2_xcpt                    := 0.U.asTypeOf(io.cpu.s2_xcpt)
  io.cpu.s2_xcpt.ma.ld              := returns && !writes && result.misaligned
  io.cpu.s2_xcpt.ma.st              := returns && writes && result.misaligned
  io.cpu.s2_xcpt.pf.ld              := returns && !writes && result.pageFault
  io.cpu.s2_xcpt.pf.st              := returns && writes && result.pageFault
  io.cpu.s2_xcpt.ae.ld              := returns && !writes && result.accessFault
  io.cpu.s2_xcpt.ae.st              := returns && writes && result.accessFault
  io.cpu.resp.valid                 := returns || probed
  io.cpu.resp.bits                  := 0.U.asTypeOf(io.cpu.resp.bits)
  io.cpu.resp.bits.tag              := command.tag
  io.cpu.resp.bits.cmd              := command.cmd
  io.cpu.resp.bits.size             := command.size
  io.cpu.resp.bits.addr             := command.addr
  io.cpu.resp.bits.signed           := command.signed
  io.cpu.resp.bits.dprv             := command.dprv
  io.cpu.resp.bits.has_data         := reads && !command.no_resp && (probed || !exception)
  io.cpu.resp.bits.data             := Mux(probed, probedLoad, result.data)
  io.cpu.resp.bits.data_word_bypass := Mux(probed, probedLoad, result.data)
  io.cpu.resp.bits.data_raw         := Mux(probed, probedLoad, result.data)
  io.cpu.resp.bits.store_data       := data
  io.cpu.replay_next                := false.B
  // A buffered completion awaits instruction replay, not a memory transaction; release ordered decoding.
  val memoryPending = state === s1 || state === s2 || state === issue || state === receive
  io.cpu.ordered       := !memoryPending
  io.cpu.store_pending := memoryPending && writes
  io.cpu.clock_enabled := true.B
  io.cpu.perf          := 0.U.asTypeOf(io.cpu.perf)
  io.cpu.perf.grant    := returns || probed
  io.cpu.perf.blocked  := !io.cpu.req.ready
  io.cpu.uncached_resp.foreach { port => port.valid := false.B; port.bits := 0.U.asTypeOf(port.bits) }
}
