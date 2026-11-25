package examples.hart.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import examples.hart.rocket.pipeline.{CpuParams, HasCpuParameters}
import freechips.rocketchip.rocket._
import freechips.rocketchip.rocket.constants.MemoryOpConstants
import framework.generic.bus.chi.rnf.CacheAtomic
import framework.hart.memory.access.{DataProbe, VirtualRequest, VirtualResponse}
import framework.hart.memory.access.configs.{Params => AccessParam}
import framework.mlp.fetch.Context

/**
 * Rocket LSU. A plain load or store probes the D$ in s1 and, on a hit, answers in s2 like
 * Rocket's own D$; a store writes the D$ in s2 when it retires. A plain access that cannot fault
 * (its TLB, region and PMP checks pass) but misses goes on through the VirtualPort without a nack;
 * a load then writes back through a replay response, as a Rocket D$ miss does. Anything else (a
 * TLB or PMP miss, an atomic, MMIO, sfence.vma or a misaligned access) is nacked in s2, performed
 * once through the VirtualPort and returned when the pipeline replays it; its effects start only
 * after both pipeline cancellation windows.
 */
@instantiable
class Lsu(cp: AccessParam)(implicit val cpuParams: CpuParams)
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
    val probe       = Flipped(new DataProbe(cp))
    val request     = Decoupled(new VirtualRequest(cp))
    val response    = Flipped(Decoupled(new VirtualResponse(cp)))
    val idle        = Output(Bool())
  })

  require(!usingHypervisor && cp.memory.tagBits <= coreParams.dcacheReqTagBits)

  class Stage extends Bundle {
    val command = new HellaCacheReq
    val pc      = UInt(vaddrBitsExtended.W)
    val pmps    = Vec(nPMPs, new PMP)
    val context = new Context
  }

  def extended(addr: UInt): UInt = if (usingVM) addr.asSInt.pad(64).asUInt else addr.pad(64)

  // The slow path: one operation through the VirtualPort, then its replayed instruction.
  val idle :: issue :: receive :: completed :: replay1 :: replay2 :: notifying :: answer :: Nil = Enum(8)
  val state                                                                                     = RegInit(idle)
  // The operation was not nacked: it answers by a replay response instead of an instruction replay.
  val detached                                                                                  = Reg(Bool())
  val command                                                                                   = Reg(new HellaCacheReq)
  val instructionPc                                                                             = Reg(UInt(vaddrBitsExtended.W))
  val data                                                                                      = Reg(UInt(64.W))
  val pmps                                                                                      = Reg(Vec(nPMPs, new PMP))
  val context                                                                                   = Reg(new Context)
  val maintaining                                                                               = RegInit(false.B)
  val result                                                                                    = Reg(new VirtualResponse(cp))
  io.capturedPmp := pmps

  // The hit path: s1 probes the D$, s2 answers.
  val s1Valid = RegInit(false.B)
  val s1      = Reg(new Stage)
  val s2Valid = RegInit(false.B)
  val s2      = Reg(new Stage)
  val s2Data  = Reg(UInt(64.W))
  val s2Probe = RegInit(false.B)
  val s2Safe  = Reg(Bool())

  val hit       = s2Valid && s2Probe && io.probe.complete.valid && io.probe.complete.bits.hit
  val fastHit   = hit && !io.cpu.s2_kill
  // An s2 miss starts the slow path when it is free and is otherwise nacked.
  val missed    = s2Valid && !io.cpu.s2_kill && !hit
  val slow      = missed && state === idle
  val detaching = slow && s2Safe

  val matches = io.cpu.req.bits.tag === command.tag && io.cpu.req.bits.addr === command.addr &&
    io.cpu.req.bits.cmd === command.cmd && io.cpu.req.bits.size === command.size && io.pc === instructionPc
  io.cpu.req.ready := state === idle || (state === completed && (!io.cpu.req.valid || matches))
  when(io.cpu.req.fire) {
    when(state === idle) {
      assert(!io.cpu.req.bits.dv && io.cpu.req.bits.dprv =/= 2.U, "LSU requires nonvirtual U, S or M privilege")
      if (io.cpu.req.bits.tag.getWidth > cp.memory.tagBits) {
        assert(
          !io.cpu.req.bits.tag(io.cpu.req.bits.tag.getWidth - 1, cp.memory.tagBits).orR,
          "LSU arbiter tag bits must be zero"
        )
      }
    }.otherwise {
      assert(matches, "LSU replay changed instruction identity")
      state := replay1
    }
  }
  s1Valid          := io.cpu.req.fire && state === idle
  when(io.cpu.req.fire && state === idle) {
    s1.command := io.cpu.req.bits
    s1.pc      := io.pc
    s1.pmps    := io.pmp
    s1.context := io.context
  }

  val s1Address = extended(s1.command.addr)
  val s1Plain   = s1.command.cmd === M_XRD || s1.command.cmd === M_XWR
  val s1Write   = s1.command.cmd === M_XWR
  val checker   = Module(new PMPChecker(3))
  checker.io.prv                  := s1.command.dprv
  checker.io.pmp                  := s1.pmps
  checker.io.addr                 := io.probe.paddr
  checker.io.size                 := s1.command.size
  io.probe.allowed                := !io.probe.paddr(63, paddrBits).orR && Mux(s1Write, checker.io.w, checker.io.r)
  io.probe.request.valid          := s1Valid && !io.cpu.s1_kill && s1Plain && state === idle && !missed
  io.probe.request.bits           := 0.U.asTypeOf(io.probe.request.bits)
  io.probe.request.bits.vaddr     := s1Address
  io.probe.request.bits.tag       := s1.command.tag(cp.memory.tagBits - 1, 0)
  io.probe.request.bits.size      := s1.command.size
  io.probe.request.bits.write     := s1Write
  io.probe.request.bits.signed    := s1.command.signed
  io.probe.request.bits.data      := io.cpu.s1_data.data
  io.probe.request.bits.privilege := s1.command.dprv
  io.probe.request.bits.sum       := s1.context.sum
  io.probe.request.bits.mxr       := s1.context.mxr
  io.probe.request.bits.satpMode  := s1.context.satp(63, 60)
  io.probe.request.bits.rootPpn   := s1.context.satp(43, 0)
  s2Valid                         := s1Valid && !io.cpu.s1_kill
  when(s1Valid) {
    s2      := s1
    s2Data  := io.cpu.s1_data.data
    s2Probe := io.probe.accepted
    s2Safe  := io.probe.translated && s1Plain && state === idle && !missed
  }

  val s2Write = s2.command.cmd === M_XWR
  val s2Reads = isRead(s2.command.cmd)
  io.probe.cancel := s2Valid && s2Probe && io.cpu.s2_kill
  io.probe.retire := s2Valid && s2Probe && !io.cpu.s2_kill

  val s2Address    = extended(s2.command.addr)
  val s2Misaligned = (s2Address & ((1.U(64.W) << s2.command.size) - 1.U)).orR
  when(slow) {
    detached          := detaching
    command           := s2.command
    instructionPc     := s2.pc
    data              := s2Data
    pmps              := s2.pmps
    context           := s2.context
    when(s2Misaligned && s2.command.cmd =/= M_SFENCE) {
      result             := 0.U.asTypeOf(result)
      result.tag         := s2.command.tag(cp.memory.tagBits - 1, 0)
      result.misaligned  := true.B
      result.accessFault := false.B
      state              := completed
    }.otherwise(state := issue)
  }

  val writes  = isWrite(command.cmd)
  val reads   = isRead(command.cmd)
  val address = extended(command.addr)
  val sfence  = command.cmd === M_SFENCE

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

  io.maintenance.valid            := state === issue && sfence
  io.maintenance.bits             := 0.U.asTypeOf(io.maintenance.bits)
  io.maintenance.bits.addr        := command.addr
  io.maintenance.bits.asid        := data
  io.maintenance.bits.rs1         := command.size(0)
  io.maintenance.bits.rs2         := command.size(1)
  io.maintained.ready             := state === receive && maintaining
  when(io.maintenance.fire) { maintaining := true.B; state := receive }
  when(io.maintained.fire) {
    assert(io.maintained.bits, "LSU maintenance must acknowledge completion")
    result      := 0.U.asTypeOf(result)
    result.tag  := command.tag(cp.memory.tagBits - 1, 0)
    maintaining := false.B
    state       := completed
  }
  io.request.valid                := state === issue && !sfence
  io.request.bits                 := 0.U.asTypeOf(io.request.bits)
  io.request.bits.vaddr           := address
  io.request.bits.tag             := command.tag(cp.memory.tagBits - 1, 0)
  io.request.bits.size            := command.size
  io.request.bits.write           := command.cmd === M_XWR
  io.request.bits.execute         := false.B
  io.request.bits.signed          := command.signed
  io.request.bits.data            := data
  io.request.bits.atomic          := atom
  io.request.bits.privilege       := command.dprv
  io.request.bits.sum             := context.sum
  io.request.bits.mxr             := context.mxr
  io.request.bits.satpMode        := context.satp(63, 60)
  io.request.bits.rootPpn         := context.satp(43, 0)
  when(io.request.fire) {
    assert(
      command.cmd === M_XRD || command.cmd === M_XWR || command.cmd === M_XLR ||
        command.cmd === M_XSC || isAMO(command.cmd),
      "LSU unsupported memory operation"
    )
    state := receive
  }
  io.response.ready               := state === receive && !maintaining
  when(io.response.fire) {
    assert(io.response.bits.tag === command.tag(cp.memory.tagBits - 1, 0), "LSU virtual response tag mismatch")
    assert(
      !detached || !(io.response.bits.misaligned || io.response.bits.pageFault || io.response.bits.accessFault),
      "LSU access that passed its checks faulted"
    )
    result := io.response.bits
    state  := Mux(detached, Mux(reads && !command.no_resp, notifying, idle), completed)
  }
  // replay_next frees the write port in the cycle before the replay response.
  when(state === notifying)(state := answer)
  when(state === answer)(state    := idle)
  when(state === replay1)(state   := Mux(io.cpu.s1_kill, completed, replay2))
  when(state === replay2)(state   := Mux(io.cpu.s2_kill, completed, idle))

  // A hit's doubleword, aligned and extended as the load asks.
  val raw = io.probe.complete.bits.value >> (s2.command.addr(2, 0) << 3)

  val load = MuxLookup(s2.command.size, raw)(Seq(
    0.U -> Cat(Fill(56, s2.command.signed && raw(7)), raw(7, 0)),
    1.U -> Cat(Fill(48, s2.command.signed && raw(15)), raw(15, 0)),
    2.U -> Cat(Fill(32, s2.command.signed && raw(31)), raw(31, 0))
  ))

  val returns   = state === replay2 && !io.cpu.s2_kill || state === answer
  val exception = result.misaligned || result.pageFault || result.accessFault
  val answered  = Mux(fastHit, s2.command, command)
  io.cpu.s2_nack           := missed && !detaching
  io.cpu.s2_nack_cause_raw := false.B
  io.cpu.s2_uncached       := false.B
  io.cpu.s2_paddr          := answered.addr(paddrBits - 1, 0)
  io.cpu.s2_gpa            := 0.U
  io.cpu.s2_gpa_is_pte     := false.B
  io.cpu.s2_xcpt           := 0.U.asTypeOf(io.cpu.s2_xcpt)
  val traps = state === replay2 && !io.cpu.s2_kill
  io.cpu.s2_xcpt.ma.ld              := traps && !writes && result.misaligned
  io.cpu.s2_xcpt.ma.st              := traps && writes && result.misaligned
  io.cpu.s2_xcpt.pf.ld              := traps && !writes && result.pageFault
  io.cpu.s2_xcpt.pf.st              := traps && writes && result.pageFault
  io.cpu.s2_xcpt.ae.ld              := traps && !writes && result.accessFault
  io.cpu.s2_xcpt.ae.st              := traps && writes && result.accessFault
  io.cpu.resp.valid                 := returns || fastHit
  io.cpu.resp.bits                  := 0.U.asTypeOf(io.cpu.resp.bits)
  io.cpu.resp.bits.tag              := answered.tag
  io.cpu.resp.bits.cmd              := answered.cmd
  io.cpu.resp.bits.size             := answered.size
  io.cpu.resp.bits.addr             := answered.addr
  io.cpu.resp.bits.signed           := answered.signed
  io.cpu.resp.bits.dprv             := answered.dprv
  io.cpu.resp.bits.has_data         := Mux(fastHit, s2Reads && !s2.command.no_resp, reads && !command.no_resp && !exception)
  io.cpu.resp.bits.data             := Mux(fastHit, load, result.data)
  io.cpu.resp.bits.data_word_bypass := io.cpu.resp.bits.data
  io.cpu.resp.bits.data_raw         := io.cpu.resp.bits.data
  io.cpu.resp.bits.store_data       := Mux(fastHit, s2Data, data)
  io.cpu.resp.bits.replay           := state === answer
  io.cpu.replay_next                := state === notifying
  // A buffered completion awaits instruction replay, not a memory transaction; release ordered decoding.
  val memoryPending = s1Valid || s2Valid || state === issue || state === receive || state === notifying ||
    state === answer
  io.idle              := state === idle && !s1Valid && !s2Valid
  io.cpu.ordered       := !memoryPending
  io.cpu.store_pending := s1Valid && isWrite(s1.command.cmd) || s2Valid && s2Write ||
    (state === issue || state === receive) && writes
  io.cpu.clock_enabled := true.B
  io.cpu.perf          := 0.U.asTypeOf(io.cpu.perf)
  io.cpu.perf.grant    := returns || fastHit
  io.cpu.perf.blocked  := !io.cpu.req.ready
  io.cpu.uncached_resp.foreach { port => port.valid := false.B; port.bits := 0.U.asTypeOf(port.bits) }
}
