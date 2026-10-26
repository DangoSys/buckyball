package examples.core.rocket

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import examples.core.rocket.configs.Params
import examples.core.rocket.pipeline.{CpuParams, HasCpuParameters, RocketBB}
import freechips.rocketchip.rocket.{CSRs, PMP, PMPChecker, PRV}
import freechips.rocketchip.tile.{CustomCSR, FPU, FPUCoreIO}
import framework.core.clink.CoreInterrupts
import framework.core.execution.fetch.Fetch
import framework.core.memory.access.{UncachedRequest, UncachedResponse, VirtualPort}
import framework.acc.frontend.RoCCIO

/** ACAL Rocket pipeline attached to the framework's memory and local accelerator protocols. */
@instantiable
class Execution(config: Params, commandsEnabled: Boolean) extends Module with HasCpuParameters {
  implicit lazy val cpuParams: CpuParams = config.cpu
  val access = config.memory.memoryAccess

  @public
  val io = IO(new Bundle {
    val interrupts   = Input(new CoreInterrupts)
    val fetch        = Flipped(new VirtualPort(access))
    val data         = Flipped(new VirtualPort(access))
    val timeRequest  = Decoupled(new UncachedRequest(access.memory))
    val timeResponse = Flipped(Decoupled(new UncachedResponse(access.memory)))
    val rocc         = Flipped(new RoCCIO(xLen, config.permission))
  })

  val pipeline: Instance[RocketBB] = Instantiate(new RocketBB(
    Seq(CustomCSR(CSRs.time, (BigInt(1) << 64) - 1, Some(BigInt(0)))),
    commandsEnabled,
    false,
    false,
    false,
    false
  ))

  val fetch: Instance[Fetch] = Instantiate(new Fetch(config.fetch))
  val lsu:   Instance[Lsu]   = Instantiate(new Lsu(access))
  val cpu = pipeline.io
  cpu.hartid                    := config.hartId.U
  cpu.reset_vector              := config.resetVector.U
  cpu.interrupts                := 0.U.asTypeOf(cpu.interrupts)
  cpu.interrupts.mtip           := io.interrupts.timer
  cpu.interrupts.msip           := io.interrupts.software
  cpu.interrupts.meip           := io.interrupts.external
  cpu.interrupts.seip.foreach(_ := io.interrupts.supervisorExternal)
  cpu.traceStall                := false.B
  cpu.ptw.perf                  := 0.U.asTypeOf(cpu.ptw.perf)
  cpu.ptw.clock_enabled         := true.B
  cpu.ptw.customCSRs.csrs.foreach { csr =>
    csr.stall := false.B
    csr.set   := false.B
    csr.sdata := 0.U
  }
  val fpu = coreParams.fpu.map(cfg => Instantiate(new FPU(cfg)))
  fpu match {
    case Some(unit) =>
      cpu.fpu :<>= unit.io.waiveAs[FPUCoreIO](_.cp_req, _.cp_resp)
      unit.io.cp_req.valid  := false.B
      unit.io.cp_req.bits   := 0.U.asTypeOf(unit.io.cp_req.bits)
      unit.io.cp_resp.ready := true.B
    case None       =>
      cpu.fpu.store_data       := 0.U
      cpu.fpu.toint_data       := 0.U
      cpu.fpu.fcsr_flags.valid := false.B
      cpu.fpu.fcsr_flags.bits  := 0.U
      cpu.fpu.fcsr_rdy         := true.B
      cpu.fpu.nack_mem         := false.B
      cpu.fpu.illegal_rm       := false.B
      cpu.fpu.dec              := 0.U.asTypeOf(cpu.fpu.dec)
      cpu.fpu.sboard_set       := false.B
      cpu.fpu.sboard_clr       := false.B
      cpu.fpu.sboard_clra      := 0.U
  }

  val satp = if (usingVM) Cat(cpu.ptw.ptbr.mode, cpu.ptw.ptbr.asid.pad(16), cpu.ptw.ptbr.ppn.pad(44)) else 0.U(64.W)
  lsu.io.cpu <> cpu.dmem
  lsu.io.pc                := cpu.dmemPc
  lsu.io.pmp               := cpu.ptw.pmp
  lsu.io.context.privilege := cpu.ptw.status.dprv
  lsu.io.context.satp      := satp
  lsu.io.context.sum       := cpu.ptw.status.sum
  lsu.io.context.mxr       := cpu.ptw.status.mxr
  // A younger memory operation cannot pass an older unaccepted accelerator command.
  val blockFresh = cpu.roccOlderPending && lsu.io.idle
  lsu.io.cpu.req.valid     := cpu.dmem.req.valid && !blockFresh
  cpu.dmem.req.ready       := lsu.io.cpu.req.ready && !blockFresh
  io.data.request <> lsu.io.request
  lsu.io.response <> io.data.response
  lsu.io.maintenance.ready := true.B
  val maintenancePending = RegInit(false.B)
  when(lsu.io.maintenance.fire)(maintenancePending := true.B)
  when(lsu.io.maintained.fire)(maintenancePending  := false.B)
  lsu.io.maintained.valid                          := maintenancePending
  lsu.io.maintained.bits                           := true.B
  io.data.flushTranslation                         := lsu.io.maintenance.fire
  io.fetch.flushTranslation                        := lsu.io.maintenance.fire

  fetch.io.resetVector       := config.resetVector.U
  fetch.io.context.privilege := cpu.ptw.status.prv
  fetch.io.context.satp      := Mux(cpu.ptw.status.prv === PRV.M.U, 0.U, satp)
  fetch.io.context.sum       := cpu.ptw.status.sum
  fetch.io.context.mxr       := cpu.ptw.status.mxr
  val redirect = if (usingVM) cpu.imem.req.bits.pc.asSInt.pad(64).asUInt else cpu.imem.req.bits.pc.pad(64)
  fetch.io.redirect.valid               := RegNext(cpu.imem.req.valid, false.B)
  fetch.io.redirect.bits                := RegEnable(redirect, cpu.imem.req.valid)
  fetch.io.flush                        := RegNext(cpu.imem.flush_icache, false.B)
  fetch.io.invalidate                   := RegNext(cpu.imem.sfence.valid, false.B)
  cpu.imem.resp.valid                   := fetch.io.packet.valid
  fetch.io.packet.ready                 := cpu.imem.resp.ready
  cpu.imem.resp.bits                    := 0.U.asTypeOf(cpu.imem.resp.bits)
  cpu.imem.resp.bits.pc                 := fetch.io.packet.bits.pc
  cpu.imem.resp.bits.data               := fetch.io.packet.bits.data
  cpu.imem.resp.bits.mask               := fetch.io.packet.bits.mask
  cpu.imem.resp.bits.xcpt.pf.inst       := fetch.io.packet.bits.pageFault
  cpu.imem.resp.bits.xcpt.ae.inst       := fetch.io.packet.bits.accessFault
  cpu.imem.npc                          := fetch.io.npc
  cpu.imem.clock_enabled                := true.B
  cpu.imem.perf                         := 0.U.asTypeOf(cpu.imem.perf)
  cpu.imem.gpa.valid                    := false.B
  cpu.imem.gpa.bits                     := 0.U
  cpu.imem.gpa_is_pte                   := false.B
  io.fetch.request.valid                := fetch.io.request.valid
  fetch.io.request.ready                := io.fetch.request.ready
  io.fetch.request.bits                 := 0.U.asTypeOf(io.fetch.request.bits)
  io.fetch.request.bits.vaddr           := fetch.io.request.bits.addr
  io.fetch.request.bits.size            := 3.U
  io.fetch.request.bits.execute         := true.B
  io.fetch.request.bits.privilege       := fetch.io.request.bits.context.privilege
  io.fetch.request.bits.sum             := fetch.io.request.bits.context.sum
  io.fetch.request.bits.mxr             := fetch.io.request.bits.context.mxr
  io.fetch.request.bits.satpMode        := fetch.io.request.bits.context.satp(63, 60)
  io.fetch.request.bits.rootPpn         := fetch.io.request.bits.context.satp(43, 0)
  fetch.io.response.valid               := io.fetch.response.valid
  io.fetch.response.ready               := fetch.io.response.ready
  fetch.io.response.bits.data           := io.fetch.response.bits.data
  fetch.io.response.bits.line           := 0.U
  fetch.io.response.bits.lineExecutable := false.B
  fetch.io.response.bits.pageFault      := io.fetch.response.bits.pageFault
  fetch.io.response.bits.accessFault    := io.fetch.response.bits.accessFault
  // Coherent I/D caches observe stores through Home; fence.i also discards fetch packets.
  fetch.io.maintenance.ready            := lsu.io.idle && !io.rocc.busy
  val fetchMaintaining = RegInit(false.B)
  when(fetch.io.maintenance.fire)(fetchMaintaining := true.B)
  when(fetch.io.maintained.fire)(fetchMaintaining  := false.B)
  fetch.io.maintained.valid                        := fetchMaintaining
  fetch.io.maintained.bits                         := true.B

  val fetchPmp = Reg(Vec(nPMPs, new PMP))
  when(io.fetch.request.fire)(fetchPmp := cpu.ptw.pmp)
  for ((port, pmps) <- Seq(io.fetch -> fetchPmp, io.data -> lsu.io.capturedPmp)) {
    val checker = Instantiate(new PMPChecker(3))
    checker.io.prv  := port.authorizationRequest.bits.privilege
    checker.io.pmp  := pmps
    checker.io.addr := port.authorizationRequest.bits.paddr
    checker.io.size := port.authorizationRequest.bits.size
    val pending = RegInit(false.B)
    val allowed = Reg(Bool())
    port.authorizationRequest.ready               := !pending
    port.authorizationResponse.valid              := pending
    port.authorizationResponse.bits               := allowed
    when(port.authorizationRequest.fire) {
      val req = port.authorizationRequest.bits
      allowed := !req.paddr(63, paddrBits).orR &&
        (!req.read || checker.io.r) && (!req.write || checker.io.w) && (!req.execute || checker.io.x)
      pending := true.B
    }
    when(port.authorizationResponse.fire)(pending := false.B)
  }

  // Freeze a custom command only on its architectural acceptance in Rocket's WB stage.
  io.rocc.cmd.valid                     := cpu.rocc.cmd.valid && cpu.roccCommitEligible && lsu.io.idle
  cpu.rocc.cmd.ready                    := io.rocc.cmd.ready && cpu.roccCommitEligible && lsu.io.idle
  io.rocc.cmd.bits.instruction.raw_inst := cpu.rocc.cmd.bits.raw_inst
  io.rocc.cmd.bits.instruction.pc       := cpu.rocc.cmd.bits.pc
  io.rocc.cmd.bits.instruction.rs1Data  := cpu.rocc.cmd.bits.rs1Data
  io.rocc.cmd.bits.instruction.rs2Data  := cpu.rocc.cmd.bits.rs2Data
  io.rocc.cmd.bits.satp                 := satp
  io.rocc.cmd.bits.effectivePrivilege   := cpu.ptw.status.dprv
  io.rocc.cmd.bits.sum                  := cpu.ptw.status.sum
  io.rocc.cmd.bits.mxr                  := cpu.ptw.status.mxr
  for ((target, source) <- io.rocc.cmd.bits.pmp.zip(cpu.ptw.pmp)) {
    target.cfg  := source.cfg.asTypeOf(target.cfg)
    target.addr := source.addr
    target.mask := source.mask
  }
  cpu.rocc.busy := io.rocc.busy
  cpu.rocc.interrupt      := io.rocc.interrupt
  cpu.rocc.resp.valid     := io.rocc.resp.valid
  cpu.rocc.resp.bits.rd   := io.rocc.resp.bits.rd
  cpu.rocc.resp.bits.data := io.rocc.resp.bits.data
  io.rocc.resp.ready      := cpu.rocc.resp.ready

  // Read the shared device timer on demand; counting never occupies the AXI bus.
  val timeIdle :: timeIssue :: timeWait :: timeReady :: Nil = Enum(4)
  val timeState                                             = RegInit(timeIdle)
  val time                                                  = cpu.rocc.csrs(0)
  when(time.ren && timeState === timeIdle)(timeState                           := timeIssue)
  io.timeRequest.valid                                                         := timeState === timeIssue
  io.timeRequest.bits                                                          := 0.U.asTypeOf(io.timeRequest.bits)
  io.timeRequest.bits.addr                                                     := config.mtimeAddress.U
  io.timeRequest.bits.size                                                     := 3.U
  when(io.timeRequest.fire)(timeState                                          := timeWait)
  io.timeResponse.ready                                                        := timeState === timeWait
  time.stall                                                                   := timeState =/= timeReady
  time.set                                                                     := io.timeResponse.fire
  time.sdata                                                                   := io.timeResponse.bits.data
  when(io.timeResponse.fire) {
    assert(!io.timeResponse.bits.error, "CLINT mtime read failed")
    timeState := timeReady
  }
  val retired = cpu.trace.insns(0)
  when(retired.valid && !retired.exception && retired.insn(6, 0) === "h73".U &&
    retired.insn(31, 20) === CSRs.time.U && timeState === timeReady)(timeState := timeIdle)
  cpu.interruptReady                                                           := lsu.io.idle && timeState === timeIdle && !io.rocc.busy
}
