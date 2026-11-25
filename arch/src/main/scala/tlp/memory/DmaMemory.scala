package framework.tlp.memory

import framework.hart.memory.{_}
import framework.generic.contract.tlink.TaskContext

import framework.generic.contract.memory.access.preflight.{Params => PreparationParams}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.hart.memory.configs.MemoryParam
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.bus.{axi, axi4}
import framework.hart.memory.access.mmu.Walker
import framework.hart.memory.access.preflight.PermissionCheck
import framework.hart.memory.access.preflight.configs.Config
import framework.generic.acc.memdomain.RangeWalker
import framework.generic.memory.interlock.MaintenanceOp
import framework.generic.memory.coherence.PageQuery
import framework.generic.dma.Address
import framework.generic.dma.Dma
import framework.generic.dma.Receive
import framework.generic.dma.Send

/** The burst of a thread stalled on a page fault, at the virtual address of the burst. */
class DmaFault extends Bundle {
  val write   = Bool()
  val address = UInt(64.W)
}

/**
 * A thread's DMA as its hart translates it. A page fault stalls the burst before it reaches DDR;
 * the owner of the thread then retries the translation (resolve true) or fails the burst (false).
 */
class TranslatedDma(p: axi4.Params) extends Dma(p) {
  val fault   = Flipped(Valid(new DmaFault))
  val resolve = Valid(Bool())
}

/**
 * The DMA of a hart's threads to the DDR. Every thread has its own channel, an AXI port of the
 * bank's width: its bursts move at the same time as the others'. The data never passes a cache;
 * the hart keeps the caches coherent with it on the control path. Before a burst moves, the hart
 * translates and authorizes it, then has the lines of its range cleaned out of the caches (a
 * read) or cleaned and invalidated (a write); while the burst is live the CPU waits on an access
 * that conflicts with it. The address phases of the channels take turns; their data does not. A
 * burst stays within one page: when the device Home holds no line of it in any cache (`page`),
 * the lines are not walked.
 *
 * A burst stays within one page, so stalling it before its address phase makes a page fault
 * precise; access faults and PMP denials fail the burst at once.
 */
@instantiable
class DmaMemory(
  bus:         axi4.Params,
  permission:  PmpParams,
  memory:      MemoryParam,
  addressBits: Int,
  backing:     Boolean,
  ports:       Int)
    extends Module {
  val access      = memory.memoryAccess
  val tracking    = memory.dcache.tracking
  val walkerParam = access.walker.copy(entries = 0)
  val ownerBits   = math.max(1, log2Ceil(ports))

  @public val io = IO(new Bundle {
    val clients  = Vec(ports, Flipped(new TranslatedDma(bus)))
    val launch   = Input(Vec(ports, Valid(new TaskContext(permission))))
    val services = new MemCtrl(tracking, walkerParam)
    val page     = new PageQuery(addressBits)
    val axi      = Vec(ports, new axi4.Port(bus))
  })

  val contexts = Reg(Vec(ports, new TaskContext(permission)))
  for (i <- 0 until ports) when(io.launch(i).valid)(contexts(i) := io.launch(i).bits)

  val rIdle :: rPending :: rIssue :: rData :: rFailed :: Nil                           = Enum(5)
  val wIdle :: wPending :: wIssue :: wData :: wResponse :: wFailed :: wComplete :: Nil = Enum(7)
  val rState                                                                           = RegInit(VecInit(Seq.fill(ports)(rIdle)))
  val wState                                                                           = RegInit(VecInit(Seq.fill(ports)(wIdle)))
  val read                                                                             = Reg(Vec(ports, new Address(bus)))
  val write                                                                            = Reg(Vec(ports, new Address(bus)))
  val readPA                                                                           = Reg(Vec(ports, UInt(bus.addressBits.W)))
  val writePA                                                                          = Reg(Vec(ports, UInt(bus.addressBits.W)))
  val readBeat                                                                         = RegInit(VecInit(Seq.fill(ports)(0.U(9.W))))
  val writeBeat                                                                        = RegInit(VecInit(Seq.fill(ports)(0.U(9.W))))
  val writeError                                                                       = RegInit(VecInit(Seq.fill(ports)(false.B)))

  // The address phase in progress: translation, permission, then the cache lines of the range.
  val idle :: translate :: translated :: authorize :: permitted :: maintain :: maintaining :: stalled :: Nil = Enum(8)
  val state                                                                                                  = RegInit(idle)
  val owner                                                                                                  = Reg(UInt(ownerBits.W))
  val next                                                                                                   = RegInit(0.U(ownerBits.W))
  val translatingWrite                                                                                       = Reg(Bool())
  val preferWrite                                                                                            = RegInit(false.B)
  val context                                                                                                = Reg(new TaskContext(permission))
  val address                                                                                                = Reg(UInt(64.W))
  val bytes                                                                                                  = Reg(UInt(13.W))
  val byteOffset                                                                                             = Reg(UInt(4.W))
  val physical                                                                                               = Reg(UInt(bus.addressBits.W))
  val walker                                                                                                 = Instantiate(new Walker(walkerParam))
  val lines                                                                                                  = Instantiate(new RangeWalker(tracking, homeRanges = false))

  val check = Instantiate(new PermissionCheck(Config(
    permission,
    PreparationParams(bus = access.memory.chi, contexts = 2, beatBytes = bus.bytes),
    access.regions
  )))

  def firstLine(base: UInt): UInt = Cat(base(bus.addressBits - 1, 6), 0.U(6.W))
  def lastLine(base:  UInt, offset: UInt, count: UInt): UInt =
    Cat((base + offset + count - 1.U)(bus.addressBits - 1, 6), 0.U(6.W))

  for (i <- 0 until ports) {
    val client   = io.clients(i)
    val port     = io.axi(i)
    val received = Instantiate(new Receive(new axi.Beat(bus.dataBits)))
    val returned = Instantiate(new Send(new axi.Beat(bus.dataBits, bus.idBits, 0, 2)))
    client.read.ready                     := rState(i) === rIdle && !reset.asBool
    client.write.ready                    := wState(i) === wIdle && !reset.asBool
    received.io.input <> client.writeData
    client.readData <> returned.io.output
    returned.io.input.valid               := rState(i) === rFailed || rState(i) === rData && port.r.valid
    returned.io.input.bits                := 0.U.asTypeOf(returned.io.input.bits)
    returned.io.input.bits.id             := read(i).id
    returned.io.input.bits.keep           := Fill(bus.bytes, 1.U(1.W))
    returned.io.input.bits.data           := Mux(rState(i) === rFailed, 0.U, port.r.bits.data)
    returned.io.input.bits.user           := Mux(rState(i) === rFailed, 2.U, port.r.bits.resp)
    returned.io.input.bits.last           := readBeat(i) +& 1.U === read(i).beats
    when(returned.io.input.fire) {
      readBeat(i)                                 := readBeat(i) + 1.U
      when(returned.io.input.bits.last)(rState(i) := rIdle)
    }
    received.io.output.ready              := (wState(i) === wData && port.w.ready) || wState(i) === wFailed
    when(received.io.output.fire) {
      assert(received.io.output.bits.last === (writeBeat(i) +& 1.U === write(i).beats), "DMA burst length mismatch")
      writeBeat(i)                                           := writeBeat(i) + 1.U
      when(writeBeat(i) +& 1.U === write(i).beats)(wState(i) := Mux(wState(i) === wFailed, wComplete, wResponse))
    }
    client.completed.valid                := wState(i) === wComplete
    client.completed.bits.id              := write(i).id
    client.completed.bits.error           := writeError(i)
    when(client.completed.fire)(wState(i) := wIdle)
    when(client.read.fire) {
      assert(client.read.bits.beats =/= 0.U && client.read.bits.beats <= 256.U &&
        client.read.bits.virtualAddress(3, 0) === 0.U &&
        client.read.bits.virtualAddress(11, 0) +& (client.read.bits.beats << 4) <= 4096.U)
      read(i) := client.read.bits; readBeat(i) := 0.U; rState(i) := rPending
    }
    when(client.write.fire) {
      assert(client.write.bits.beats =/= 0.U && client.write.bits.beats <= 256.U &&
        client.write.bits.virtualAddress(3, 0) === 0.U &&
        client.write.bits.virtualAddress(11, 0) +& (client.write.bits.beats << 4) <= 4096.U)
      write(i) := client.write.bits; writeBeat(i) := 0.U; writeError(i) := false.B; wState(i) := wPending
    }

    port.ar.valid                := rState(i) === rIssue
    port.ar.bits                 := 0.U.asTypeOf(port.ar.bits)
    port.ar.bits.addr            := readPA(i)
    port.ar.bits.id              := read(i).id
    port.ar.bits.len             := read(i).beats - 1.U
    port.ar.bits.size            := 4.U; port.ar.bits.burst := 1.U
    when(port.ar.fire)(rState(i) := rData)
    port.r.ready                 := rState(i) === rData && returned.io.input.ready
    when(port.r.fire)(assert(port.r.bits.last === (readBeat(i) +& 1.U === read(i).beats), "DDR read length mismatch"))
    port.aw.valid                := wState(i) === wIssue
    port.aw.bits                 := 0.U.asTypeOf(port.aw.bits)
    port.aw.bits.addr            := writePA(i)
    port.aw.bits.id              := write(i).id
    port.aw.bits.len             := write(i).beats - 1.U
    port.aw.bits.size            := 4.U; port.aw.bits.burst := 1.U
    when(port.aw.fire)(wState(i) := wData)
    port.w.valid                 := wState(i) === wData && received.io.output.valid
    port.w.bits.data             := received.io.output.bits.data
    port.w.bits.strb             := received.io.output.bits.keep
    port.w.bits.last             := received.io.output.bits.last
    port.b.ready                 := wState(i) === wResponse
    when(port.b.fire) { writeError(i) := port.b.bits.resp =/= 0.U; wState(i) := wComplete }

    client.fault.valid           := state === stalled && owner === i.U
    client.fault.bits.write      := translatingWrite
    client.fault.bits.address    := address
    when(client.resolve.valid)(assert(client.fault.valid, "DMA fault resolved without a stalled burst"))
  }

  // The channels' address phases in turn; a channel's read and write alternate.
  val waiting  = VecInit((0 until ports).map(i => rState(i) === rPending || wState(i) === wPending))
  val later    = VecInit((0 until ports).map(i => waiting(i) && i.U >= next)).asUInt
  val selected = PriorityEncoder(Mux(later.orR, later, waiting.asUInt))
  when(state === idle && waiting.asUInt.orR) {
    val store = wState(selected) === wPending && (rState(selected) =/= rPending || preferWrite)
    owner            := selected; next := Mux(selected === (ports - 1).U, 0.U, selected + 1.U)
    translatingWrite := store; context := contexts(selected)
    address          := Mux(store, write(selected).virtualAddress, read(selected).virtualAddress)
    bytes            := Mux(store, write(selected).bytes, read(selected).bytes)
    byteOffset       := Mux(store, write(selected).offset, read(selected).offset)
    preferWrite      := !store; state  := translate
  }

  def fail(): Unit = {
    when(translatingWrite) { writeError(owner) := true.B; wState(owner) := wFailed }.otherwise(rState(owner) := rFailed)
    state := idle
  }

  walker.io.flush                := false.B
  walker.io.lookup.config        := 0.U.asTypeOf(walker.io.lookup.config)
  walker.io.lookup.req           := 0.U.asTypeOf(walker.io.lookup.req)
  walker.io.config.mode          := context.satp(63, 60)
  walker.io.config.rootPpn       := context.satp(43, 0)
  walker.io.req.valid            := state === translate
  walker.io.req.bits.vaddr       := address
  walker.io.req.bits.write       := translatingWrite
  walker.io.req.bits.execute     := false.B
  walker.io.req.bits.privilege   := context.effectivePrivilege
  walker.io.req.bits.sum         := context.sum
  walker.io.req.bits.mxr         := context.mxr
  when(walker.io.req.fire)(state := translated)
  walker.io.resp.ready           := state === translated
  when(walker.io.resp.fire) {
    when(walker.io.resp.bits.accessFault)(fail())
      .elsewhen(walker.io.resp.bits.pageFault)(state := stalled)
      .otherwise { physical := walker.io.resp.bits.paddr; state := authorize }
  }
  val resolution = io.clients(owner).resolve
  when(state === stalled && resolution.valid) {
    when(resolution.bits)(state := translate).otherwise(fail())
  }

  // The lines of the authorized range leave the caches before its data moves; a page with no
  // cached line has none to leave.
  io.page.address := physical
  val absent = state === maintain && !io.page.resident
  when(absent) {
    when(translatingWrite) { writePA(owner) := physical; wState(owner) := wIssue }
      .otherwise { readPA(owner) := physical; rState(owner) := rIssue }
    state := idle
  }
  lines.io.range.valid := (if (backing) state === maintain && io.page.resident else false.B)
  lines.io.range.bits.tag       := 0.U
  lines.io.range.bits.firstLine := firstLine(physical)
  lines.io.range.bits.lastLine  := lastLine(physical, byteOffset, bytes)
  lines.io.range.bits.op        := Mux(translatingWrite, MaintenanceOp.CleanInvalidate.U, MaintenanceOp.Clean.U)
  val rangeAccepted  = Wire(Bool())
  val rangeCompleted = Wire(Bool())
  val rangeOkay      = Wire(Bool())
  if (backing) {
    io.services.maintenance <> lines.io.maintenance
    lines.io.maintained <> io.services.maintained
    lines.io.done.ready := state === maintaining
    rangeAccepted       := lines.io.range.fire
    rangeCompleted      := lines.io.done.fire
    rangeOkay           := lines.io.done.bits.ok
  } else {
    io.services.maintenance.valid := state === maintain && io.page.resident && io.services.drained
    io.services.maintenance.bits  := lines.io.range.bits
    io.services.maintained.ready  := state === maintaining
    lines.io.maintenance.ready    := false.B
    lines.io.maintained.valid     := false.B
    lines.io.maintained.bits      := 0.U.asTypeOf(lines.io.maintained.bits)
    lines.io.done.ready           := false.B
    rangeAccepted                 := io.services.maintenance.fire
    rangeCompleted                := io.services.maintained.fire
    rangeOkay                     := io.services.maintained.bits.ok
  }
  when(rangeAccepted)(state := maintaining)
  when(rangeCompleted) {
    when(!rangeOkay)(fail()).otherwise {
      when(translatingWrite) { writePA(owner) := physical; wState(owner) := wIssue }
        .otherwise { readPA(owner) := physical; rState(owner) := rIssue }
      state := idle
    }
  }
  io.services.lineAddress := lines.io.lineAddress
  io.services.lineWrite := lines.io.lineWrite
  lines.io.lineMaintain := io.services.lineMaintain
  lines.io.drained      := io.services.drained

  // The CPU waits on an access to a line a live burst writes, and on a write to a line one reads.
  val cpuLine = firstLine(io.services.cpuQuery.paddr)
  def within(base: UInt, offset: UInt, count: UInt): Bool =
    cpuLine >= firstLine(base) && cpuLine <= lastLine(base, offset, count)

  val conflicts = (0 until ports).map { i =>
    ((rState(i) === rIssue || rState(i) === rData) && io.services.cpuQuery.write &&
      within(readPA(i), read(i).offset, read(i).bytes)) ||
    ((wState(i) === wIssue || wState(i) === wData || wState(i) === wResponse) &&
      within(writePA(i), write(i).offset, write(i).bytes))
  } :+ ((state === maintain || state === maintaining) && (translatingWrite || io.services.cpuQuery.write) &&
    within(physical, byteOffset, bytes))

  io.services.cpuAllow := !conflicts.reduce(_ || _)

  val pteIdle :: pteAsk :: pteWait :: pteIssue :: pteReturn :: pteReject :: pteRejectReturn :: Nil = Enum(7)
  val pteState                                                                                     = RegInit(pteIdle)
  when(pteState === pteIdle && walker.io.access.valid)(pteState := pteAsk)
  check.io.request.valid                                        := pteState === pteAsk || state === authorize
  check.io.request.bits.id                                      := 0.U
  check.io.request.bits.isPte                                   := pteState === pteAsk
  check.io.request.bits.pa                                      := Mux(pteState === pteAsk, walker.io.access.bits.addr, physical + byteOffset)
  check.io.request.bits.bytes                                   := Mux(pteState === pteAsk, 8.U, bytes)
  check.io.request.bits.write                                   := pteState =/= pteAsk && translatingWrite
  check.io.request.bits.privilege                               := Mux(pteState === pteAsk, 1.U, context.effectivePrivilege)
  check.io.contextValid                                         := true.B
  check.io.contextId                                            := 0.U
  check.io.pmp                                                  := context.pmp
  when(check.io.request.fire) {
    when(pteState === pteAsk)(pteState := pteWait).otherwise(state := permitted)
  }
  check.io.response.ready                                       := pteState === pteWait || state === permitted
  when(check.io.response.fire) {
    when(pteState === pteWait)(pteState := Mux(check.io.response.bits.allow, pteIssue, pteReject))
      .elsewhen(check.io.response.bits.allow)(state := maintain)
      .otherwise(fail())
  }
  io.services.pteRequest.valid                                  := pteState === pteIssue && walker.io.access.valid
  io.services.pteRequest.bits                                   := walker.io.access.bits
  walker.io.access.ready                                        := pteState === pteReject || pteState === pteIssue && io.services.pteRequest.ready
  when(walker.io.access.fire)(pteState                          := Mux(pteState === pteReject, pteRejectReturn, pteReturn))
  walker.io.result.valid                                        := pteState === pteRejectReturn || pteState === pteReturn && io.services.pteResponse.valid
  walker.io.result.bits.data                                    := Mux(pteState === pteRejectReturn, 0.U, io.services.pteResponse.bits.data)
  walker.io.result.bits.error                                   := pteState === pteRejectReturn || io.services.pteResponse.bits.error
  io.services.pteResponse.ready                                 := pteState === pteReturn && walker.io.result.ready
  when(walker.io.result.fire)(pteState                          := pteIdle)
}
