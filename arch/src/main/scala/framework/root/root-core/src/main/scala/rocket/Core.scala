package hier.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.system.core.rocket.{CpuParams, HasCpuParameters}
import freechips.rocketchip.tile.{CustomCSR, FPU, FPUCoreIO, TraceBundle}
import freechips.rocketchip.rocket.{CSRs, PMPChecker, PRV}
import framework.system.core.rocket.{RoCCCommandBB, RoCCResponseBB, RocketBB}
import memcore.memory.fetch.{Fetch, Params => FetchParams}
import memcore.memory.cpu.{CpuMemParams, PhysicalRegion, VirtualMemory}
import memcore.bus.chi.RequesterPort
import memcore.bus.chi.rnf.{BankedChiCache, CacheAccess, CacheAtomic, CacheResult, RnfParams}
import memcore.memory.interlock.{
  CpuQuery,
  Dispatch,
  Tag,
  Acknowledgement,
  Maintenance => MaintenanceRange,
  Params => TrackingParams
}

case class Commands(compute: Boolean, scheduler: Boolean, tracking: TrackingParams = TrackingParams()) {
  require(compute || scheduler)
}

class CommandSnapshot(tracking: TrackingParams, pmps: Int)(implicit val cpuParams: CpuParams)
    extends Bundle
    with HasCpuParameters {
  val tag                = UInt(tracking.idBits.W)
  val instruction        = new RoCCCommandBB
  val satp               = UInt(64.W)
  val effectivePrivilege = UInt(2.W)
  val sum                = Bool()
  val mxr                = Bool()
  val pmp                = Vec(pmps, new freechips.rocketchip.rocket.PMP)
}

class AdmissionPorts(tracking: TrackingParams, pmps: Int, bus: memcore.bus.chi.Params)(implicit val cpuParams: CpuParams)
    extends Bundle
    with HasCpuParameters {
  val reserve       = Decoupled(new Dispatch(tracking))
  val command       = Decoupled(new CommandSnapshot(tracking, pmps))
  val complete      = Flipped(Decoupled(new Tag(tracking)))
  val cancelled     = Flipped(Decoupled(new Tag(tracking)))
  val response      = Flipped(Decoupled(new RoCCResponseBB))
  val interrupt     = Input(Bool())
  val maintenance   = Flipped(Decoupled(new MaintenanceRange(tracking)))
  val maintained    = Decoupled(new Acknowledgement(tracking))
  val cpuQuery      = Output(new CpuQuery(tracking))
  val cpuAllow      = Input(Bool())
  val cpuProbeAllow = Input(Bool())
  val pteRequest    = Flipped(Decoupled(new CacheAccess(bus)))
  val pteResponse   = Decoupled(new CacheResult)
  // Ownership registration only; TaskController must also join actual accelerator/DMA idle.
  val outstanding   = Output(UInt(log2Ceil(tracking.entries + 1).W))
}

/** One integer Rocket, own instruction/data memory, and explicit physical permission checks. */
@instantiable
class Core(
  config:                 RnfParams,
  instruction:            RnfParams,
  regions:                Seq[PhysicalRegion],
  commands:               Option[Commands] = None
)(
  implicit val cpuParams: CpuParams)
    extends Module
    with HasCpuParameters {
  override def desiredName: String = "RocketCore"
  require(!usingHypervisor && usingCompressed && fetchBytes == 4)
  require(!usingFPU || fLen <= 64, "Core memory data path supports floating-point formats up to 64 bits")
  require(instruction.chi == config.chi && instruction.nodeId != config.nodeId)
  private val cp = CpuMemParams(config.chi, tagBits = 6)

  @public
  val io = IO(new Bundle {
    val resetVector                 = Input(UInt(64.W))
    val time                        = Input(UInt(64.W))
    val timerInterrupt              = Input(Bool())
    val softwareInterrupt           = Input(Bool())
    val externalInterrupt           = Input(Bool())
    val supervisorExternalInterrupt = Input(Bool())
    val hartId                      = Input(UInt(hartIdLen.W))
    val chi                         = new RequesterPort(config.chi)
    val instructionChi              = new RequesterPort(config.chi)
    val uncachedRequest             = Decoupled(new memcore.memory.cpu.UncachedRequest(cp))
    val uncachedResponse            = Flipped(Decoupled(new memcore.memory.cpu.UncachedResponse(cp)))
    val trace                       = Output(new TraceBundle)
    val retired                     = Output(Bool())
    val retiredPc                   = Output(UInt(64.W))
    val trapped                     = Output(Bool())
    val trapCause                   = Output(UInt(64.W))
    val trapValue                   = Output(UInt(64.W))
    val trapPc                      = Output(UInt(64.W))
    val cancelledData               = Output(Bool())
    val admission                   = commands.map(mode => new AdmissionPorts(mode.tracking, nPMPs, config.chi))
  })

  val cpu: Instance[RocketBB] =
    Instantiate(new RocketBB(
      Seq(CustomCSR(CSRs.time, (BigInt(1) << 64) - 1, Some(BigInt(0)))),
      commands.exists(_.compute),
      commands.exists(_.scheduler),
      false,
      false,
      false
    ))

  val fetch:             Instance[Fetch]          = Instantiate(new Fetch(FetchParams()))
  val lsu:               Instance[Lsu]            = Instantiate(new Lsu(cp))
  val system:            Instance[VirtualMemory]  = Instantiate(new VirtualMemory(cp, regions, 0, 16))
  // Instruction words have their own translation, PMP check and coherent L1, apart from the data broker.
  val instructionSystem: Instance[VirtualMemory]  =
    Instantiate(new VirtualMemory(cp, regions, 512, 16))
  val instructionCache:  Instance[BankedChiCache] = Instantiate(new BankedChiCache(instruction.copy(lineResult = true)))

  private val cacheTracking = commands match {
    case Some(mode) => mode.tracking
    case None       => TrackingParams(addressBits = config.chi.addressBits)
  }

  val cache: Instance[Cache] = Instantiate(new Cache(config.copy(probe = true), cacheTracking))
  val maintenanceActive     = RegInit(false.B)
  val maintenanceBlock      = WireDefault(false.B)
  val cacheOffer            = RegInit(false.B)
  val cacheOwner            = RegInit(false.B)
  val offerExternal         = RegInit(false.B)
  val ownerExternal         = RegInit(false.B)
  val offerPacket           = Reg(new CacheAccess(config.chi))
  val preferExternal        = RegInit(false.B)
  val uncachedOffer         = RegInit(false.B)
  val uncachedOwner         = RegInit(false.B)
  val uncachedOfferExternal = RegInit(false.B)
  val uncachedOwnerExternal = RegInit(false.B)
  val uncachedPacket        = Reg(new memcore.memory.cpu.UncachedRequest(cp))
  val pteDenied             = RegInit(false.B)
  commands match {
    case Some(_) =>
      val port = io.admission.get
      cache.io.maintenance <> port.maintenance
      port.maintained <> cache.io.maintained
      maintenanceBlock                              := port.maintenance.valid || maintenanceActive
      when(port.maintenance.fire)(maintenanceActive := true.B)
      when(port.maintained.fire)(maintenanceActive  := false.B)
    case None    =>
      cache.io.maintenance.valid := false.B
      cache.io.maintenance.bits  := 0.U.asTypeOf(cache.io.maintenance.bits)
      cache.io.maintained.ready  := true.B
  }
  cache.io.access <> system.io.cacheRequest
  system.io.cacheResponse <> cache.io.result
  io.chi <> cache.io.chi
  system.io.active := true.B
  // SFENCE.VMA invalidates the cached data and instruction translations.
  system.io.flushTranslation            := lsu.io.maintenance.fire
  instructionSystem.io.flushTranslation := lsu.io.maintenance.fire
  io.uncachedRequest <> system.io.uncachedRequest
  system.io.uncachedResponse <> io.uncachedResponse
  io.trace                              := cpu.io.trace
  io.retired                            := cpu.io.trace.insns(0).valid && !cpu.io.trace.insns(0).exception
  io.retiredPc                          := cpu.io.trace.insns(0).iaddr
  io.trapped                            := cpu.io.trace.insns(0).valid && cpu.io.trace.insns(0).exception
  io.trapCause                          := cpu.io.trace.insns(0).cause
  io.trapValue                          := cpu.io.trace.insns(0).tval
  io.trapPc                             := cpu.io.trace.insns(0).iaddr
  io.cancelledData                      := lsu.io.cancelled
  assert(io.resetVector(63, paddrBits) === 0.U, "Core reset vector exceeds physical width")
  cpu.io.hartid                         := io.hartId
  cpu.io.reset_vector                   := io.resetVector(paddrBits - 1, 0)
  cpu.io.interrupts                     := 0.U.asTypeOf(cpu.io.interrupts)
  cpu.io.interrupts.mtip                := io.timerInterrupt
  cpu.io.interrupts.msip                := io.softwareInterrupt
  cpu.io.interrupts.meip                := io.externalInterrupt
  cpu.io.interrupts.seip.foreach(_      := io.supervisorExternalInterrupt)
  cpu.io.traceStall                     := false.B
  // TIME samples the chip's CLINT clock; CSRFile enforces its read-only address and counter permissions.
  cpu.io.rocc.csrs(0).stall             := false.B
  cpu.io.rocc.csrs(0).set               := true.B
  cpu.io.rocc.csrs(0).sdata             := io.time
  cpu.io.ptw.perf                       := 0.U.asTypeOf(cpu.io.ptw.perf)
  cpu.io.ptw.clock_enabled              := true.B
  cpu.io.ptw.customCSRs.csrs.foreach { csr =>
    csr.stall := false.B
    csr.set   := false.B
    csr.sdata := 0.U
  }
  val fpu = coreParams.fpu.map(cfg => Module(new FPU(cfg)))
  fpu match {
    case Some(unit) =>
      cpu.io.fpu :<>= unit.io.waiveAs[FPUCoreIO](_.cp_req, _.cp_resp)
      unit.io.cp_req.valid  := false.B
      unit.io.cp_req.bits   := DontCare
      unit.io.cp_resp.ready := false.B
    case None       =>
      cpu.io.fpu.store_data       := 0.U
      cpu.io.fpu.toint_data       := 0.U
      cpu.io.fpu.fcsr_flags.valid := false.B
      cpu.io.fpu.fcsr_flags.bits  := 0.U
      cpu.io.fpu.fcsr_rdy         := true.B
      cpu.io.fpu.nack_mem         := false.B
      cpu.io.fpu.illegal_rm       := false.B
      cpu.io.fpu.dec              := 0.U.asTypeOf(cpu.io.fpu.dec)
      cpu.io.fpu.sboard_set       := false.B
      cpu.io.fpu.sboard_clr       := false.B
      cpu.io.fpu.sboard_clra      := 0.U
  }
  cpu.io.rocc.cmd.ready                 := true.B
  cpu.io.rocc.resp.valid                := false.B
  cpu.io.rocc.resp.bits                 := 0.U.asTypeOf(cpu.io.rocc.resp.bits)
  cpu.io.rocc.busy                      := false.B
  cpu.io.rocc.interrupt                 := false.B
  cpu.io.rocc.mem                       := DontCare

  lsu.io.cpu <> cpu.io.dmem
  if (commands.nonEmpty) {
    // Existing LSU owners, including the replay that retires them, must keep progressing.
    val blockFresh = (cpu.io.roccOlderPending || maintenanceBlock) && lsu.io.idle
    lsu.io.cpu.req.valid  := cpu.io.dmem.req.valid && !blockFresh
    cpu.io.dmem.req.ready := lsu.io.cpu.req.ready && !blockFresh
  }
  lsu.io.pc  := cpu.io.dmemPc
  lsu.io.pmp := cpu.io.ptw.pmp
  val satp =
    if (usingVM) Cat(cpu.io.ptw.ptbr.mode, cpu.io.ptw.ptbr.asid.pad(16), cpu.io.ptw.ptbr.ppn.pad(44)) else 0.U(64.W)
  lsu.io.context.privilege   := cpu.io.ptw.status.dprv
  lsu.io.context.satp        := satp
  lsu.io.context.sum         := cpu.io.ptw.status.sum
  lsu.io.context.mxr         := cpu.io.ptw.status.mxr
  fetch.io.resetVector       := io.resetVector
  fetch.io.context.privilege := cpu.io.ptw.status.prv
  fetch.io.context.satp      := satp
  fetch.io.context.sum       := cpu.io.ptw.status.sum
  fetch.io.context.mxr       := cpu.io.ptw.status.mxr

  // Rocket kills its IBuf on the correction cycle. Fetch receives that captured
  // correction one cycle later, breaking the BTB valid/redirect feedback path.
  val redirectedPc =
    if (usingVM) cpu.io.imem.req.bits.pc.asSInt.pad(64).asUInt
    else cpu.io.imem.req.bits.pc.pad(64)

  fetch.io.redirect.valid            := RegNext(cpu.io.imem.req.valid, false.B)
  fetch.io.redirect.bits             := RegEnable(redirectedPc, cpu.io.imem.req.valid)
  fetch.io.flush                     := RegNext(cpu.io.imem.flush_icache, false.B)
  cpu.io.imem.resp.valid             := fetch.io.packet.valid
  fetch.io.packet.ready              := cpu.io.imem.resp.ready
  cpu.io.imem.resp.bits              := 0.U.asTypeOf(cpu.io.imem.resp.bits)
  cpu.io.imem.resp.bits.pc           := fetch.io.packet.bits.pc
  cpu.io.imem.resp.bits.data         := fetch.io.packet.bits.data
  cpu.io.imem.resp.bits.mask         := fetch.io.packet.bits.mask
  cpu.io.imem.resp.bits.xcpt.pf.inst := fetch.io.packet.bits.pageFault
  cpu.io.imem.resp.bits.xcpt.ae.inst := fetch.io.packet.bits.accessFault
  cpu.io.imem.npc                    := fetch.io.npc
  cpu.io.imem.clock_enabled          := true.B
  cpu.io.imem.perf                   := 0.U.asTypeOf(cpu.io.imem.perf)
  cpu.io.imem.gpa.valid              := false.B
  cpu.io.imem.gpa.bits               := 0.U
  cpu.io.imem.gpa_is_pte             := false.B

  // Fetch prepares its context one edge before its word offer becomes visible.
  val previousPmp = RegNext(cpu.io.ptw.pmp)
  fetch.io.invalidate := RegNext(cpu.io.imem.sfence.valid, false.B) || (cpu.io.ptw.pmp.asUInt =/= previousPmp.asUInt)
  val instructionLineExecutable = RegInit(false.B)
  val fetchOffer                = RegNext(fetch.io.request.valid, false.B)
  val fetchPmp                  = Reg(Vec(nPMPs, new freechips.rocketchip.rocket.PMP))
  when(fetch.io.request.valid && !fetchOffer)(fetchPmp := previousPmp)
  val wordPmp = Mux(fetchOffer, fetchPmp, previousPmp)

  instructionSystem.io.active                 := true.B
  instructionSystem.io.request.valid          := fetch.io.request.valid
  instructionSystem.io.request.bits           := 0.U.asTypeOf(instructionSystem.io.request.bits)
  instructionSystem.io.request.bits.vaddr     := fetch.io.request.bits.addr
  instructionSystem.io.request.bits.size      := 3.U
  instructionSystem.io.request.bits.execute   := true.B
  instructionSystem.io.request.bits.privilege := fetch.io.request.bits.context.privilege
  instructionSystem.io.request.bits.sum       := fetch.io.request.bits.context.sum
  instructionSystem.io.request.bits.mxr       := fetch.io.request.bits.context.mxr
  instructionSystem.io.request.bits.satpMode  := fetch.io.request.bits.context.satp(63, 60)
  instructionSystem.io.request.bits.rootPpn   := fetch.io.request.bits.context.satp(43, 0)
  fetch.io.request.ready                      := instructionSystem.io.request.ready
  fetch.io.response.valid                     := instructionSystem.io.response.valid
  fetch.io.response.bits.data                 := instructionSystem.io.response.bits.data
  fetch.io.response.bits.line                 := instructionSystem.io.response.bits.line
  fetch.io.response.bits.lineExecutable       := instructionLineExecutable
  fetch.io.response.bits.pageFault            := instructionSystem.io.response.bits.pageFault
  fetch.io.response.bits.accessFault          := instructionSystem.io.response.bits.accessFault
  instructionSystem.io.response.ready         := fetch.io.response.ready
  when(instructionSystem.io.response.valid) {
    assert(!instructionSystem.io.response.bits.misaligned, "Aligned instruction word returned misalignment")
  }
  instructionCache.io.access <> instructionSystem.io.cacheRequest
  instructionSystem.io.cacheResponse <> instructionCache.io.result
  io.instructionChi <> instructionCache.io.chi

  val instructionPmp               = Reg(Vec(nPMPs, new freechips.rocketchip.rocket.PMP))
  val instructionAuthorization     = Module(new PMPChecker(3))
  val instructionLineAuthorization = Module(new PMPChecker(6))
  instructionLineAuthorization.io.pmp := instructionPmp
  instructionLineAuthorization.io.prv := instructionSystem.io.authorizationRequest.bits.privilege
  val instructionPhysicalLine = (instructionSystem.io.authorizationRequest.bits.paddr >> 6) << 6
  instructionLineAuthorization.io.addr := instructionPhysicalLine(paddrBits - 1, 0)
  instructionLineAuthorization.io.size := 6.U

  val instructionLineRegion = regions.filter(r => r.cacheable && r.normal && r.executable).map(r =>
    instructionPhysicalLine.pad(65) >= r.base.U(65.W) &&
      instructionPhysicalLine.pad(65) + 63.U < (r.base + r.bytes).U(65.W)
  ).foldLeft(false.B)(_ || _)

  val instructionPending = RegInit(false.B)
  val instructionAllowed = Reg(Bool())
  when(instructionSystem.io.request.fire)(instructionPmp                   := wordPmp)
  instructionAuthorization.io.pmp                                          := instructionPmp
  instructionAuthorization.io.prv                                          := instructionSystem.io.authorizationRequest.bits.privilege
  instructionAuthorization.io.addr                                         := instructionSystem.io.authorizationRequest.bits.paddr(paddrBits - 1, 0)
  instructionAuthorization.io.size                                         := instructionSystem.io.authorizationRequest.bits.size
  instructionSystem.io.authorizationRequest.ready                          := !instructionPending
  instructionSystem.io.authorizationResponse.valid                         := instructionPending
  instructionSystem.io.authorizationResponse.bits                          := instructionAllowed
  when(instructionSystem.io.authorizationRequest.fire) {
    val r = instructionSystem.io.authorizationRequest.bits
    instructionAllowed        := !r.paddr(63, paddrBits).orR &&
      (!r.read || instructionAuthorization.io.r) && (!r.execute || instructionAuthorization.io.x)
    instructionLineExecutable := !r.paddr(
      63,
      paddrBits
    ).orR && instructionLineAuthorization.io.x && instructionLineRegion
    instructionPending        := true.B
  }
  when(instructionSystem.io.authorizationResponse.fire)(instructionPending := false.B)

  // Instructions are fetched only through the cacheable L1; an uncached executable word faults.
  val instructionUncached    = RegInit(false.B)
  val instructionUncachedTag = Reg(UInt(cp.tagBits.W))
  instructionSystem.io.uncachedRequest.ready                           := !instructionUncached
  instructionSystem.io.uncachedResponse.valid                          := instructionUncached
  instructionSystem.io.uncachedResponse.bits.tag                       := instructionUncachedTag
  instructionSystem.io.uncachedResponse.bits.data                      := 0.U
  instructionSystem.io.uncachedResponse.bits.error                     := true.B
  when(instructionSystem.io.uncachedRequest.fire) {
    instructionUncached    := true.B
    instructionUncachedTag := instructionSystem.io.uncachedRequest.bits.tag
  }
  when(instructionSystem.io.uncachedResponse.fire)(instructionUncached := false.B)

  val idle :: data :: barrier :: translationBarrier :: Nil = Enum(4)
  val state                                                = RegInit(idle)
  cpu.io.interruptReady := state === idle && lsu.io.idle && !cpu.io.dmem.req.valid
  val authorizationPmp = Reg(Vec(nPMPs, new freechips.rocketchip.rocket.PMP))
  val authorization    = Module(new PMPChecker(3))
  authorization.io.pmp  := authorizationPmp
  authorization.io.prv  := system.io.authorizationRequest.bits.privilege
  authorization.io.addr := system.io.authorizationRequest.bits.paddr(paddrBits - 1, 0)
  authorization.io.size := system.io.authorizationRequest.bits.size
  val authorizationPending = RegInit(false.B)
  val authorizationAllowed = Reg(Bool())
  system.io.authorizationRequest.ready                            := !authorizationPending
  system.io.authorizationResponse.valid                           := authorizationPending
  system.io.authorizationResponse.bits                            := authorizationAllowed
  when(system.io.authorizationRequest.fire) {
    val r = system.io.authorizationRequest.bits
    authorizationAllowed := !r.paddr(63, paddrBits).orR &&
      (!r.read || authorization.io.r) && (!r.write || authorization.io.w) &&
      (!r.execute || authorization.io.x)
    authorizationPending := true.B
  }
  when(system.io.authorizationResponse.fire)(authorizationPending := false.B)

  // Data fast path: the LSU's plain access completes in s2 when a data-TLB translation (or
  // identity), permission, region, NPU interlock and an L1 hit all agree while the slow broker
  // path is fully idle. A TLB miss or fault leaves the access to the broker, which walks.
  val probe = lsu.io.probe
  system.io.lookup.config.mode       := probe.req.bits.satp(63, 60)
  system.io.lookup.config.rootPpn    := probe.req.bits.satp(43, 0)
  system.io.lookup.req.vaddr         := probe.req.bits.addr
  system.io.lookup.req.write         := probe.req.bits.write
  system.io.lookup.req.execute       := false.B
  system.io.lookup.req.privilege     := probe.req.bits.privilege
  system.io.lookup.req.sum           := probe.req.bits.sum
  system.io.lookup.req.mxr           := probe.req.bits.mxr
  instructionSystem.io.lookup.config := 0.U.asTypeOf(instructionSystem.io.lookup.config)
  instructionSystem.io.lookup.req    := 0.U.asTypeOf(instructionSystem.io.lookup.req)
  val probeAddr  = system.io.lookup.paddr.pad(64)
  val probeBytes = (1.U(4.W) << probe.req.bits.size)(3, 0)
  val probeLast  = probeAddr.pad(65) + probeBytes - 1.U

  def probeRegion(
    write: Boolean
  ): Bool = regions.filter(r => r.cacheable && r.normal && (if (write) r.writable else r.readable)).map(r =>
    probeAddr.pad(65) >= r.base.U(65.W) && probeLast < (r.base + r.bytes).U(65.W)
  )
    .foldLeft(false.B)(_ || _)

  val probePmp = Module(new PMPChecker(3))
  probePmp.io.pmp  := lsu.io.capturedPmp
  probePmp.io.prv  := probe.req.bits.privilege
  probePmp.io.addr := probeAddr(paddrBits - 1, 0)
  probePmp.io.size := probe.req.bits.size
  val probePermitted = system.io.lookup.hit && !probeAddr(63, paddrBits).orR &&
    Mux(probe.req.bits.write, probePmp.io.w && probeRegion(true), probePmp.io.r && probeRegion(false))
  val slowIdle       = state === idle && !authorizationPending && !cacheOffer && !cacheOwner &&
    !uncachedOffer && !uncachedOwner && !pteDenied && !maintenanceBlock
  val probeQuery     = probe.active && probePermitted && slowIdle
  val probeAllow     = WireDefault(true.B)
  val l1Probe        = cache.io.probe.get
  val probeAllowed   = probePermitted && slowIdle && probeAllow
  l1Probe.req.valid         := probe.req.valid && probeAllowed
  l1Probe.req.bits.addr     := probeAddr(config.chi.addressBits - 1, 0)
  l1Probe.req.bits.write    := probe.req.bits.write
  l1Probe.req.bits.data     := probe.req.bits.data << (probeAddr(2, 0) << 3)
  l1Probe.req.bits.mask     := (MuxLookup(probe.req.bits.size, 255.U(8.W))(Seq(
    0.U -> 1.U(8.W),
    1.U -> 3.U(8.W),
    2.U -> 15.U(8.W)
  )) <<
    probeAddr(2, 0))(7, 0)
  probe.req.ready           := l1Probe.req.ready && probeAllowed
  l1Probe.cancel            := probe.cancel || (probe.active && !probeAllowed)
  l1Probe.retire            := probe.retire && probeAllowed && !probe.cancel
  probe.cancelled           := l1Probe.cancel
  probe.complete.valid      := l1Probe.complete.valid && probeAllowed && !probe.cancel
  probe.complete.bits.hit   := l1Probe.complete.bits.hit
  probe.complete.bits.value := l1Probe.complete.bits.value
  l1Probe.complete.ready    := probe.complete.ready && probeAllowed && !probe.cancel
  // Broker state covers the full accepted VM/PTE/data/fetch request through its response.
  // The wrapper adds L1 outstanding==0; CMO activity/ACK must not gate its own drain.
  val originalIdleDrain    = state === idle && lsu.io.idle
  val blockedPhysicalQuery = WireDefault(false.B)
  cache.io.olderRequestsDrained := (originalIdleDrain || blockedPhysicalQuery) && !probe.active &&
    !authorizationPending && !cacheOffer && !cacheOwner && !uncachedOffer && !uncachedOwner && !pteDenied
  commands.foreach { mode =>
    val port         = io.admission.get
    val tracking     = mode.tracking
    require(tracking.addressBits == config.chi.addressBits)
    require(tracking.idBits >= log2Ceil(tracking.entries))
    val live         = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
    val sent         = RegInit(VecInit(Seq.fill(tracking.entries)(false.B)))
    val pending      = RegInit(false.B)
    val snapshot     = Reg(new CommandSnapshot(tracking, nPMPs))
    val free         = VecInit(live.map(!_)).asUInt
    val selected     = PriorityEncoder(free)
    val olderDrained = state === idle && lsu.io.idle && !authorizationPending &&
      cache.io.outstanding === 0.U && !cacheOffer && !cacheOwner && !uncachedOffer && !uncachedOwner && !pteDenied
    val capacity     = free.orR && !pending && olderDrained && !maintenanceBlock
    // Interlock ready is independent of valid. Pulse reserve only for an actual atomic WB acceptance;
    // Rocket's pre-acceptance WB replay may withdraw its offer and must not create an orphan reservation.
    cpu.io.rocc.cmd.ready := capacity && port.reserve.ready &&
      (!cpu.io.rocc.cmd.valid || cpu.io.roccCommitEligible)
    port.reserve.valid    := cpu.io.rocc.cmd.valid && cpu.io.roccCommitEligible && capacity && port.reserve.ready
    port.reserve.bits.id  := selected
    assert(cpu.io.rocc.cmd.fire === port.reserve.fire, "Admission reservation must match Rocket command acceptance")
    port.command.valid    := pending
    port.command.bits     := snapshot
    when(port.reserve.fire) {
      snapshot.tag                := selected
      snapshot.instruction        := cpu.io.rocc.cmd.bits
      snapshot.satp               := satp
      // Like BEMU, an M-mode command translates through an enabled satp as S-mode.
      snapshot.effectivePrivilege := Mux(
        cpu.io.ptw.status.dprv === PRV.M.U && satp(63, 60) =/= 0.U,
        PRV.S.U,
        cpu.io.ptw.status.dprv
      )
      snapshot.sum                := cpu.io.ptw.status.sum
      snapshot.mxr                := cpu.io.ptw.status.mxr
      snapshot.pmp                := cpu.io.ptw.pmp
      live(selected)              := true.B
      sent(selected)              := false.B
      pending                     := true.B
    }
    val tagIndexBits = log2Ceil(tracking.entries)
    def index(tag: UInt): UInt = tag(tagIndexBits - 1, 0)
    when(port.command.fire) { pending := false.B; sent(index(snapshot.tag)) := true.B }
    val completeFound = port.complete.bits.tag < tracking.entries.U && live(index(port.complete.bits.tag)) && sent(
      index(port.complete.bits.tag)
    )
    val cancelledFound = port.cancelled.bits.tag < tracking.entries.U && live(index(port.cancelled.bits.tag)) && sent(
      index(port.cancelled.bits.tag)
    )
    port.complete.ready   := completeFound
    port.cancelled.ready  := cancelledFound && !(port.complete.valid && port.complete.bits.tag === port.cancelled.bits.tag)
    when(port.complete.valid)(assert(completeFound, "Admission completion must identify a delivered live command"))
    when(port.cancelled.valid) {
      assert(cancelledFound, "Admission cancellation must identify a delivered live command")
      assert(
        !port.complete.valid || port.complete.bits.tag =/= port.cancelled.bits.tag,
        "Admission cannot complete and cancel the same command"
      )
    }
    when(port.complete.fire) {
      live(index(port.complete.bits.tag)) := false.B; sent(index(port.complete.bits.tag)) := false.B
    }
    when(port.cancelled.fire) {
      live(index(port.cancelled.bits.tag)) := false.B; sent(index(port.cancelled.bits.tag)) := false.B
    }
    port.outstanding      := PopCount(live)
    cpu.io.rocc.resp <> port.response
    cpu.io.rocc.interrupt := port.interrupt
    // Older accepted broker/PTW/LSU owners continue. Fresh EX memory and fresh word offers
    // are stopped above/below until the MEM/WB command has reserved its identity.
    val access   = system.io.cacheRequest.bits
    val uncached = system.io.uncachedRequest
    port.cpuQuery.valid                := (system.io.cacheRequest.valid && access.atomic =/= CacheAtomic.Fence.U) ||
      uncached.valid || probeQuery
    port.cpuQuery.paddr                := Mux(
      probeQuery,
      probeAddr(config.chi.addressBits - 1, 0),
      Mux(uncached.valid, uncached.bits.addr(config.chi.addressBits - 1, 0), access.addr)
    )
    port.cpuQuery.sizeLog2             := Mux(
      probeQuery,
      probe.req.bits.size,
      Mux(uncached.valid, uncached.bits.size, Mux(access.atomicWord, 2.U, 3.U))
    )
    port.cpuQuery.write                := Mux(
      probeQuery,
      probe.req.bits.write,
      Mux(
        uncached.valid,
        uncached.bits.write,
        access.write || access.atomic === CacheAtomic.SC.U ||
          (access.atomic >= CacheAtomic.Swap.U && access.atomic <= CacheAtomic.MaxU.U)
      )
    )
    probeAllow                         := port.cpuProbeAllow
    port.cpuQuery.olderDispatchPending := false.B
    // Reservation itself drained all old owners; an unissued blocked physical
    // query is younger and must not hold up the reservation's maintenance.
    blockedPhysicalQuery               := (system.io.cacheRequest.valid || uncached.valid) && !port.cpuAllow
    val legalPte       = !port.pteRequest.bits.write && port.pteRequest.bits.atomic === CacheAtomic.None.U &&
      !port.pteRequest.bits.atomicWord && port.pteRequest.bits.addr(2, 0) === 0.U &&
      port.pteRequest.bits.mask === 0.U
    when(port.pteRequest.valid) {
      assert(legalPte, "Core external PTE request must be an aligned read with the Walker mask")
    }
    val systemEligible = system.io.cacheRequest.valid && port.cpuAllow &&
      (!maintenanceBlock || state === data)
    // Preparation already authorized this PA using its retained command context.
    // Region classification additionally forbids device reads, including PTE reads.
    val pteAddr        = port.pteRequest.bits.addr.pad(65)
    val pteNormal      = regions.filter(r => r.normal && r.readable).map(r =>
      pteAddr >= r.base.U(65.W) && pteAddr + 8.U <= (r.base + r.bytes).U(65.W)
    ).foldLeft(false.B)(_ || _)
    val pteCached      = regions.filter(r => r.normal && r.readable && r.cacheable).map(r =>
      pteAddr >= r.base.U(65.W) && pteAddr + 8.U <= (r.base + r.bytes).U(65.W)
    ).foldLeft(false.B)(_ || _)
    val pteBusy        = (cacheOffer && offerExternal) || (cacheOwner && ownerExternal) ||
      (uncachedOffer && uncachedOfferExternal) || (uncachedOwner && uncachedOwnerExternal) || pteDenied
    val pteAvailable   = port.pteRequest.valid && legalPte && !maintenanceBlock && !pteBusy
    val pteEligible    = pteAvailable && pteCached
    val selectPte      = pteEligible && (!systemEligible || preferExternal)
    when(!cacheOffer && !cacheOwner && (systemEligible || pteEligible)) {
      offerPacket   := Mux(selectPte, port.pteRequest.bits, access)
      offerExternal := selectPte
      cacheOffer    := true.B
    }
    cache.io.access.valid := cacheOffer
    cache.io.access.bits                  := offerPacket
    system.io.cacheRequest.ready          := cacheOffer && !offerExternal && cache.io.access.ready
    port.pteRequest.ready                 := cacheOffer && offerExternal && cache.io.access.ready
    when(cache.io.access.fire) {
      cacheOffer     := false.B
      cacheOwner     := true.B
      ownerExternal  := offerExternal
      preferExternal := !offerExternal
    }
    system.io.cacheResponse.valid         := cacheOwner && !ownerExternal && cache.io.result.valid
    system.io.cacheResponse.bits          := cache.io.result.bits
    port.pteResponse.valid                := cacheOwner && ownerExternal && cache.io.result.valid
    port.pteResponse.bits                 := cache.io.result.bits
    cache.io.result.ready                 := cacheOwner && Mux(ownerExternal, port.pteResponse.ready, system.io.cacheResponse.ready)
    when(cache.io.result.fire)(cacheOwner := false.B)
    val uncachedPteEligible = pteAvailable && pteNormal && !pteCached
    val uncachedCpuEligible = uncached.valid && port.cpuAllow &&
      (!maintenanceBlock || state === data)
    val selectUncachedPte   = uncachedPteEligible && (!uncachedCpuEligible || preferExternal)
    when(!uncachedOffer && !uncachedOwner && (uncachedCpuEligible || uncachedPteEligible)) {
      uncachedPacket        := uncached.bits
      when(selectUncachedPte) {
        uncachedPacket.addr   := port.pteRequest.bits.addr
        uncachedPacket.tag    := 0.U
        uncachedPacket.size   := 3.U
        uncachedPacket.write  := false.B
        uncachedPacket.data   := 0.U
        uncachedPacket.atomic := CacheAtomic.None.U
        uncachedPacket.normal := true.B
      }
      uncachedOfferExternal := selectUncachedPte
      uncachedOffer         := true.B
    }
    io.uncachedRequest.valid := uncachedOffer
    io.uncachedRequest.bits                      := uncachedPacket
    uncached.ready                               := uncachedOffer && !uncachedOfferExternal && io.uncachedRequest.ready
    when(uncachedOffer && uncachedOfferExternal) {
      port.pteRequest.ready := io.uncachedRequest.ready
    }
    when(io.uncachedRequest.fire) {
      uncachedOffer         := false.B
      uncachedOwner         := true.B
      uncachedOwnerExternal := uncachedOfferExternal
      preferExternal        := !uncachedOfferExternal
    }
    system.io.uncachedResponse.valid             := uncachedOwner && !uncachedOwnerExternal && io.uncachedResponse.valid
    system.io.uncachedResponse.bits              := io.uncachedResponse.bits
    io.uncachedResponse.ready                    := uncachedOwner &&
      Mux(uncachedOwnerExternal, port.pteResponse.ready, system.io.uncachedResponse.ready)
    when(uncachedOwner && uncachedOwnerExternal) {
      port.pteResponse.valid      := io.uncachedResponse.valid
      port.pteResponse.bits.data  := Mux(io.uncachedResponse.bits.error, 0.U, io.uncachedResponse.bits.data)
      port.pteResponse.bits.error := io.uncachedResponse.bits.error
    }
    when(io.uncachedResponse.valid) {
      assert(uncachedOwner, "Core uncached response requires an accepted owner")
      assert(io.uncachedResponse.bits.tag === uncachedPacket.tag, "Core uncached response tag must match its owner")
    }
    when(io.uncachedResponse.fire)(uncachedOwner := false.B)
    when(pteAvailable && !pteNormal) {
      port.pteRequest.ready                := true.B
      when(port.pteRequest.fire)(pteDenied := true.B)
    }
    when(pteDenied) {
      port.pteResponse.valid                := true.B
      port.pteResponse.bits.data            := 0.U
      port.pteResponse.bits.error           := true.B
      when(port.pteResponse.fire)(pteDenied := false.B)
    }
  }

  lsu.io.maintenance.ready                                       := false.B
  lsu.io.maintained.valid                                        := state === translationBarrier
  lsu.io.maintained.bits                                         := true.B
  lsu.io.request.ready                                           := false.B
  lsu.io.response.valid                                          := false.B
  lsu.io.response.bits                                           := system.io.response.bits
  system.io.request.valid                                        := false.B
  system.io.request.bits                                         := lsu.io.request.bits
  system.io.response.ready                                       := false.B
  fetch.io.maintenance.ready                                     := false.B
  fetch.io.maintained.valid                                      := state === barrier
  fetch.io.maintained.bits                                       := true.B
  when(state === idle) {
    when(lsu.io.maintenance.valid && cache.io.outstanding === 0.U) {
      // No translation cache exists: draining the prior broker transaction is the SFENCE boundary.
      lsu.io.maintenance.ready            := true.B
      when(lsu.io.maintenance.fire)(state := translationBarrier)
    }.elsewhen(lsu.io.request.valid) {
      system.io.request.valid := true.B
      lsu.io.request.ready    := system.io.request.ready
      when(system.io.request.fire) {
        authorizationPmp := lsu.io.capturedPmp
        state            := data
      }
    }.elsewhen(fetch.io.maintenance.valid && lsu.io.idle && cache.io.outstanding === 0.U) {
      // Drained stores own their lines in the data L1; the coherent instruction L1 is snooped on those writes.
      fetch.io.maintenance.ready            := true.B
      when(fetch.io.maintenance.fire)(state := barrier)
    }
  }
  when(state === data) {
    lsu.io.response.valid               := system.io.response.valid
    system.io.response.ready            := lsu.io.response.ready
    when(system.io.response.fire)(state := idle)
  }
  when(fetch.io.maintained.fire || lsu.io.maintained.fire)(state := idle)
}
