package framework.core.memory

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.core.configs.CoreParam
import framework.thread.ant.TaskContext
import framework.thread.tlink.{Address, Dma, Receive, Send}
import framework.ip.bus.{axi, axi4}
import framework.core.memory.access.mmu.Walker
import framework.core.memory.access.preflight.{PermissionCheck, Params => PreparationParams}
import framework.core.memory.access.preflight.configs.Config

/** Core-owned translation for external thread bursts, after local dataflow selection. */
@instantiable
class DmaMemory(core: CoreParam) extends Module {
  val ports       = core.threads.size
  val bus         = core.interconnect.bus
  val permission  = core.privileged.permission
  val access      = core.memory.memoryAccess
  val walkerParam = access.walker.copy(entries = 0)
  val ownerBits   = math.max(1, log2Ceil(ports))

  @public val io = IO(new Bundle {
    val clients  = Vec(ports, Flipped(new Dma(bus)))
    val launch   = Input(Vec(ports, Valid(new TaskContext(permission))))
    val services = new MemCtrl(core.memory.dcache.tracking, walkerParam)
    val axi      = new axi4.Port(bus)
  })

  val contexts = Reg(Vec(ports, new TaskContext(permission)))
  for (i <- 0 until ports) when(io.launch(i).valid)(contexts(i) := io.launch(i).bits)

  val rIdle :: rPending :: rIssue :: rData :: rFailed :: Nil                           = Enum(5)
  val wIdle :: wPending :: wIssue :: wData :: wResponse :: wFailed :: wComplete :: Nil = Enum(7)
  val rState                                                                           = RegInit(rIdle)
  val wState                                                                           = RegInit(wIdle)
  val read                                                                             = Reg(new Address(bus))
  val write                                                                            = Reg(new Address(bus))
  val readOwner                                                                        = Reg(UInt(ownerBits.W))
  val writeOwner                                                                       = Reg(UInt(ownerBits.W))
  val readNext                                                                         = RegInit(0.U(ownerBits.W))
  val writeNext                                                                        = RegInit(0.U(ownerBits.W))
  val readContext                                                                      = Reg(new TaskContext(permission))
  val writeContext                                                                     = Reg(new TaskContext(permission))
  val readPA                                                                           = Reg(UInt(bus.addressBits.W))
  val writePA                                                                          = Reg(UInt(bus.addressBits.W))
  val readBeat                                                                         = RegInit(0.U(9.W))
  val writeBeat                                                                        = RegInit(0.U(9.W))
  val writeError                                                                       = RegInit(false.B)

  val readCandidates  = VecInit(io.clients.map(_.read.valid)).asUInt
  val writeCandidates = VecInit(io.clients.map(_.write.valid)).asUInt
  val readLater       = VecInit(io.clients.zipWithIndex.map { case (c, i) => c.read.valid && i.U >= readNext }).asUInt
  val writeLater      = VecInit(io.clients.zipWithIndex.map { case (c, i) => c.write.valid && i.U >= writeNext }).asUInt
  val selectedRead    = PriorityEncoder(Mux(readLater.orR, readLater, readCandidates))
  val selectedWrite   = PriorityEncoder(Mux(writeLater.orR, writeLater, writeCandidates))
  val received        = (0 until ports).map(_ => Instantiate(new Receive(new axi.Beat(bus.dataBits))))
  val returned        = (0 until ports).map(_ => Instantiate(new Send(new axi.Beat(bus.dataBits, bus.idBits, 0, 2))))

  for (i <- 0 until ports) {
    val client = io.clients(i)
    client.read.ready                  := rState === rIdle && selectedRead === i.U && !reset.asBool
    client.write.ready                 := wState === wIdle && selectedWrite === i.U && !reset.asBool
    received(i).io.input <> client.writeData
    client.readData <> returned(i).io.output
    returned(i).io.input.valid         := readOwner === i.U && (rState === rFailed || rState === rData && io.axi.r.valid)
    returned(i).io.input.bits          := 0.U.asTypeOf(returned(i).io.input.bits)
    returned(i).io.input.bits.id       := read.id
    returned(i).io.input.bits.keep     := Fill(bus.bytes, 1.U(1.W))
    returned(i).io.input.bits.data     := Mux(rState === rFailed, 0.U, io.axi.r.bits.data)
    returned(i).io.input.bits.user     := Mux(rState === rFailed, 2.U, io.axi.r.bits.resp)
    returned(i).io.input.bits.last     := readBeat +& 1.U === read.beats
    when(returned(i).io.input.fire) {
      readBeat                                    := readBeat + 1.U
      when(returned(i).io.input.bits.last)(rState := rIdle)
    }
    received(i).io.output.ready        := writeOwner === i.U &&
      ((wState === wData && io.axi.w.ready) || wState === wFailed)
    when(received(i).io.output.fire) {
      assert(received(i).io.output.bits.last === (writeBeat +& 1.U === write.beats), "DMA burst length mismatch")
      writeBeat                                     := writeBeat + 1.U
      when(writeBeat +& 1.U === write.beats)(wState := Mux(wState === wFailed, wComplete, wResponse))
    }
    client.completed.valid             := wState === wComplete && writeOwner === i.U
    client.completed.bits.id           := write.id
    client.completed.bits.error        := writeError
    when(client.completed.fire)(wState := wIdle)
    when(client.read.fire) {
      assert(client.read.bits.beats =/= 0.U && client.read.bits.beats <= 256.U &&
        client.read.bits.virtualAddress(3, 0) === 0.U &&
        client.read.bits.virtualAddress(11, 0) +& (client.read.bits.beats << 4) <= 4096.U)
      read     := client.read.bits; readOwner := i.U; readContext := contexts(i); readBeat := 0.U
      readNext := (if (i + 1 == ports) 0 else i + 1).U
      rState   := rPending
    }
    when(client.write.fire) {
      assert(client.write.bits.beats =/= 0.U && client.write.bits.beats <= 256.U &&
        client.write.bits.virtualAddress(3, 0) === 0.U &&
        client.write.bits.virtualAddress(11, 0) +& (client.write.bits.beats << 4) <= 4096.U)
      write      := client.write.bits; writeOwner := i.U; writeContext := contexts(i); writeBeat := 0.U;
      writeError := false.B
      writeNext  := (if (i + 1 == ports) 0 else i + 1).U
      wState     := wPending
    }
  }

  val idle :: translate :: translated :: authorize :: permitted :: Nil = Enum(5)
  val state                                                            = RegInit(idle)
  val translatingWrite                                                 = Reg(Bool())
  val preferWrite                                                      = RegInit(false.B)
  val context                                                          = Reg(new TaskContext(permission))
  val address                                                          = Reg(UInt(64.W))
  val bytes                                                            = Reg(UInt(13.W))
  val byteOffset                                                       = Reg(UInt(4.W))
  val physical                                                         = Reg(UInt(bus.addressBits.W))
  val walker                                                           = Instantiate(new Walker(walkerParam))

  val check = Instantiate(new PermissionCheck(Config(
    permission,
    PreparationParams(bus = access.memory.chi, contexts = 2, beatBytes = bus.bytes),
    access.regions
  )))

  when(state === idle && (rState === rPending || wState === wPending)) {
    val store = wState === wPending && (rState =/= rPending || preferWrite)
    translatingWrite := store; context := Mux(store, writeContext, readContext)
    address          := Mux(store, write.virtualAddress, read.virtualAddress)
    bytes            := Mux(store, write.bytes, read.bytes)
    byteOffset       := Mux(store, write.offset, read.offset)
    preferWrite      := !store; state  := translate
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
    when(walker.io.resp.bits.pageFault || walker.io.resp.bits.accessFault) {
      when(translatingWrite) { writeError := true.B; wState := wFailed }.otherwise(rState := rFailed)
      state := idle
    }.otherwise { physical := walker.io.resp.bits.paddr; state := authorize }
  }

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
      .otherwise {
        when(translatingWrite) {
          writePA    := physical
          writeError := !check.io.response.bits.allow
          wState     := Mux(check.io.response.bits.allow, wIssue, wFailed)
        }.otherwise {
          readPA := physical
          rState := Mux(check.io.response.bits.allow, rIssue, rFailed)
        }
        state := idle
      }
  }
  io.services.cpuAllow                                          := true.B
  io.services.maintenance.valid                                 := false.B
  io.services.maintenance.bits                                  := 0.U.asTypeOf(io.services.maintenance.bits)
  io.services.maintained.ready                                  := true.B
  io.services.pteRequest.valid                                  := pteState === pteIssue && walker.io.access.valid
  io.services.pteRequest.bits                                   := walker.io.access.bits
  walker.io.access.ready                                        := pteState === pteReject || pteState === pteIssue && io.services.pteRequest.ready
  when(walker.io.access.fire)(pteState                          := Mux(pteState === pteReject, pteRejectReturn, pteReturn))
  walker.io.result.valid                                        := pteState === pteRejectReturn || pteState === pteReturn && io.services.pteResponse.valid
  walker.io.result.bits.data                                    := Mux(pteState === pteRejectReturn, 0.U, io.services.pteResponse.bits.data)
  walker.io.result.bits.error                                   := pteState === pteRejectReturn || io.services.pteResponse.bits.error
  io.services.pteResponse.ready                                 := pteState === pteReturn && walker.io.result.ready
  when(walker.io.result.fire)(pteState                          := pteIdle)

  io.axi.ar.valid             := rState === rIssue
  io.axi.ar.bits              := 0.U.asTypeOf(io.axi.ar.bits)
  io.axi.ar.bits.addr         := readPA
  io.axi.ar.bits.id           := read.id
  io.axi.ar.bits.len          := read.beats - 1.U
  io.axi.ar.bits.size         := 4.U; io.axi.ar.bits.burst := 1.U; io.axi.ar.bits.cache := 15.U
  when(io.axi.ar.fire)(rState := rData)
  io.axi.r.ready              := rState === rData && VecInit(returned.map(_.io.input.ready))(readOwner)
  when(io.axi.r.fire)(assert(io.axi.r.bits.last === (readBeat +& 1.U === read.beats), "DDR read length mismatch"))
  io.axi.aw.valid             := wState === wIssue
  io.axi.aw.bits              := 0.U.asTypeOf(io.axi.aw.bits)
  io.axi.aw.bits.addr         := writePA
  io.axi.aw.bits.id           := write.id
  io.axi.aw.bits.len          := write.beats - 1.U
  io.axi.aw.bits.size         := 4.U; io.axi.aw.bits.burst := 1.U; io.axi.aw.bits.cache := 15.U
  when(io.axi.aw.fire)(wState := wData)
  val data = VecInit(received.map(_.io.output.bits))(writeOwner)
  io.axi.w.valid     := wState === wData && VecInit(received.map(_.io.output.valid))(writeOwner)
  io.axi.w.bits.data := data.data
  io.axi.w.bits.strb := data.keep
  io.axi.w.bits.last := data.last
  io.axi.b.ready     := wState === wResponse
  when(io.axi.b.fire) { writeError := io.axi.b.bits.resp =/= 0.U; wState := wComplete }
}
