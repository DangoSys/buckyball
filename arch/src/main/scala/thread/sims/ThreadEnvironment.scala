package framework.thread.sims

import _root_.circt.stage.ChiselStage
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.device.peripherals.scu.SCUWriteDPI
import framework.device.sims.verilator.{BBSimDRAM, BdbClkDPI}
import framework.generic.bus.{axi, axi4}
import framework.generic.contract.memory.access.preflight.configs.PmpParams
import framework.generic.contract.sims.configs.SimParam
import framework.generic.contract.tlink.{TLinkParam, Task}
import framework.generic.dma.{Dma, Receive, Send}
import framework.generic.memory.spm
import framework.thread.configs.ThreadParam
import framework.thread.tbus.ThreadFactory
import framework.thread.tlink.TLinkIO
import framework.tlp.tss.Store
import framework.tlp.tss.configs.{Params => StoreParams}

/** The thread's DMA to the ideal memory: one read and one write burst live, at the addresses the thread gives. */
@instantiable
class IdealDma(p: axi4.Params) extends Module {

  @public val io = IO(new Bundle {
    val dma = Flipped(new Dma(p))
    val axi = new axi4.Port(p)
  })

  val reading    = RegInit(false.B)
  val readIssued = RegInit(false.B)
  val read       = Reg(new framework.generic.dma.Address(p))
  io.dma.read.ready               := !reading && !reset.asBool
  when(io.dma.read.fire) { read := io.dma.read.bits; reading := true.B; readIssued := false.B }
  io.axi.ar.valid                 := reading && !readIssued
  io.axi.ar.bits                  := 0.U.asTypeOf(io.axi.ar.bits)
  io.axi.ar.bits.id               := read.id
  io.axi.ar.bits.addr             := read.virtualAddress
  io.axi.ar.bits.len              := read.beats - 1.U
  io.axi.ar.bits.size             := log2Ceil(p.bytes).U
  io.axi.ar.bits.burst            := 1.U
  when(io.axi.ar.fire)(readIssued := true.B)
  val returned = Instantiate(new Send(new axi.Beat(p.dataBits, p.idBits, 0, 2)))
  returned.io.input.valid                                    := io.axi.r.valid
  returned.io.input.bits                                     := 0.U.asTypeOf(returned.io.input.bits)
  returned.io.input.bits.tid                                 := read.id
  returned.io.input.bits.tkeep                               := Fill(p.bytes, 1.U(1.W))
  returned.io.input.bits.tdata                               := io.axi.r.bits.data
  returned.io.input.bits.tuser                               := io.axi.r.bits.resp
  returned.io.input.bits.tlast                               := io.axi.r.bits.last
  io.axi.r.ready                                             := returned.io.input.ready
  io.dma.readData <> returned.io.output
  when(returned.io.input.fire && io.axi.r.bits.last)(reading := false.B)

  val writing     = RegInit(false.B)
  val writeIssued = RegInit(false.B)
  val write       = Reg(new framework.generic.dma.Address(p))
  io.dma.write.ready               := !writing && !reset.asBool
  when(io.dma.write.fire) { write := io.dma.write.bits; writing := true.B; writeIssued := false.B }
  io.axi.aw.valid                  := writing && !writeIssued
  io.axi.aw.bits                   := 0.U.asTypeOf(io.axi.aw.bits)
  io.axi.aw.bits.id                := write.id
  io.axi.aw.bits.addr              := write.virtualAddress
  io.axi.aw.bits.len               := write.beats - 1.U
  io.axi.aw.bits.size              := log2Ceil(p.bytes).U
  io.axi.aw.bits.burst             := 1.U
  when(io.axi.aw.fire)(writeIssued := true.B)
  val received = Instantiate(new Receive(new axi.Beat(p.dataBits)))
  received.io.input <> io.dma.writeData
  io.axi.w.valid              := writing && writeIssued && received.io.output.valid
  io.axi.w.bits.data          := received.io.output.bits.data
  io.axi.w.bits.strb          := received.io.output.bits.keep
  io.axi.w.bits.last          := received.io.output.bits.last
  received.io.output.ready    := writing && writeIssued && io.axi.w.ready
  io.dma.completed.valid      := io.axi.b.valid
  io.dma.completed.bits.id    := write.id
  io.dma.completed.bits.error := io.axi.b.bits.resp =/= 0.U
  io.axi.b.ready              := io.dma.completed.ready
  when(io.axi.b.fire)(writing := false.B)
}

/** An AXI slave on a store: one beat a store access, a read or a write burst at a time. */
@instantiable
class StoreSlave(p: axi4.Params, s: spm.Params) extends Module {
  require(p.dataBits == s.dataBits)

  @public val io = IO(new Bundle {
    val axi   = Flipped(new axi4.Port(p))
    val store = new spm.Port(s)
  })

  val idle :: reading :: readWait :: writing :: writeWait :: response :: Nil = Enum(6)
  val state                                                                  = RegInit(idle)
  val address                                                                = Reg(UInt(p.addressBits.W))
  val id                                                                     = Reg(UInt(p.idBits.W))
  val remaining                                                              = Reg(UInt(8.W))
  val last                                                                   = Reg(Bool())
  val error                                                                  = Reg(Bool())

  io.axi.ar.ready               := state === idle
  io.axi.aw.ready               := state === idle && !io.axi.ar.valid
  when(io.axi.ar.fire) {
    address := io.axi.ar.bits.addr; id := io.axi.ar.bits.id; remaining := io.axi.ar.bits.len; state := reading
  }
  when(io.axi.aw.fire) {
    address := io.axi.aw.bits.addr; id := io.axi.aw.bits.id; error := false.B; state := writing
  }
  io.store.request.valid        := state === reading || (state === writing && io.axi.w.valid)
  io.store.request.bits.address := address
  io.store.request.bits.size    := s.laneBits.U
  io.store.request.bits.write   := state === writing
  io.store.request.bits.data    := io.axi.w.bits.data
  io.store.request.bits.mask    := Mux(state === writing, io.axi.w.bits.strb, 0.U)
  io.axi.w.ready                := state === writing && io.store.request.ready
  when(io.store.request.fire) {
    last  := io.axi.w.bits.last
    state := Mux(state === reading, readWait, writeWait)
  }
  io.store.response.ready       := (state === readWait && io.axi.r.ready) || state === writeWait
  io.axi.r.valid                := state === readWait && io.store.response.valid
  io.axi.r.bits.id              := id
  io.axi.r.bits.data            := io.store.response.bits.data
  io.axi.r.bits.resp            := Mux(io.store.response.bits.error, 2.U, 0.U)
  io.axi.r.bits.last            := remaining === 0.U
  when(io.axi.r.fire) {
    address   := address + p.bytes.U
    remaining := remaining - 1.U
    state     := Mux(remaining === 0.U, idle, reading)
  }
  when(state === writeWait && io.store.response.fire) {
    error   := error || io.store.response.bits.error
    address := address + p.bytes.U
    state   := Mux(last, response, writing)
  }
  io.axi.b.valid                := state === response
  io.axi.b.bits.id              := id
  io.axi.b.bits.resp            := Mux(error, 2.U, 0.U)
  when(io.axi.b.fire)(state     := idle)
}

/**
 * The hart side of a thread alone: it loads the program image at the start of the ideal memory into the thread
 * through the TSS, launches it once and reports its completion value. The image is a header of 16 bytes, the code
 * bytes and the TLS image bytes, then the code and the TLS image, each a multiple of 16 bytes.
 */
@instantiable
class Host(link: TLinkParam, permission: PmpParams, memoryBase: BigInt) extends Module {

  @public val io = IO(new Bundle {
    val axi  = new axi4.Port(link.dma)
    val tss  = new spm.Port(link.tss)
    val task = Flipped(new Task(link.taskBits, permission))
    val exit = Output(Valid(UInt(64.W)))
  })

  val (header :: headerWait :: fetch :: fetchWait :: store :: storeWait :: transfer :: transferWait :: start :: run ::
    done :: Nil) = Enum(11)
  val state      = RegInit(header)
  val codeBytes  = Reg(UInt(32.W))
  val tlsBytes   = Reg(UInt(32.W))
  val code       = RegInit(true.B)
  val offset     = Reg(UInt(32.W))
  val beat       = Reg(UInt(link.tss.dataBits.W))
  val value      = Reg(UInt(64.W))
  val bytes      = Mux(code, codeBytes, tlsBytes)
  val source     = memoryBase.U + 16.U + Mux(code, 0.U, codeBytes) + offset

  io.axi.ar.valid            := state === header || state === fetch
  io.axi.ar.bits             := 0.U.asTypeOf(io.axi.ar.bits)
  io.axi.ar.bits.addr        := Mux(state === header, memoryBase.U, source)
  io.axi.ar.bits.size        := log2Ceil(link.dma.bytes).U
  io.axi.ar.bits.burst       := 1.U
  io.axi.r.ready             := state === headerWait || state === fetchWait
  io.axi.aw.valid            := false.B
  io.axi.aw.bits             := 0.U.asTypeOf(io.axi.aw.bits)
  io.axi.w.valid             := false.B
  io.axi.w.bits              := 0.U.asTypeOf(io.axi.w.bits)
  io.axi.b.ready             := false.B
  when(io.axi.ar.fire)(state := Mux(state === header, headerWait, fetchWait))
  when(io.axi.r.fire) {
    assert(io.axi.r.bits.resp === 0.U && io.axi.r.bits.last, "Thread image read failed")
    when(state === headerWait) {
      codeBytes := io.axi.r.bits.data(31, 0)
      tlsBytes  := io.axi.r.bits.data(95, 64)
      offset    := 0.U
      state     := fetch
    }.otherwise {
      beat  := io.axi.r.bits.data
      state := store
    }
  }

  io.tss.request.valid            := state === store
  io.tss.request.bits.address     := link.tss.base.U + offset
  io.tss.request.bits.size        := link.tss.laneBits.U
  io.tss.request.bits.write       := true.B
  io.tss.request.bits.data        := beat
  io.tss.request.bits.mask        := Fill(link.tss.beatBytes, 1.U(1.W))
  io.tss.response.ready           := state === storeWait
  when(io.tss.request.fire)(state := storeWait)
  when(io.tss.response.fire) {
    assert(!io.tss.response.bits.error, "Thread image store failed")
    offset := offset + link.tss.beatBytes.U
    state  := Mux(offset + link.tss.beatBytes.U === bytes, transfer, fetch)
  }

  io.task.transfer.valid             := state === transfer
  io.task.transfer.bits.code         := code
  io.task.transfer.bits.toLocal      := true.B
  io.task.transfer.bits.localOffset  := 0.U
  io.task.transfer.bits.sharedOffset := 0.U
  io.task.transfer.bits.bytes        := bytes
  io.task.transferred.ready          := state === transferWait
  when(io.task.transfer.fire)(state  := transferWait)
  when(io.task.transferred.fire) {
    offset := 0.U
    code   := false.B
    state  := Mux(code && tlsBytes =/= 0.U, fetch, start)
  }

  io.task.inUse                                 := true.B
  io.task.cancel                                := false.B
  io.task.start.valid                           := state === start
  io.task.start.bits                            := 0.U.asTypeOf(io.task.start.bits)
  io.task.start.bits.descriptor.task            := 1.U
  io.task.start.bits.descriptor.codeEnd         := codeBytes
  io.task.start.bits.descriptor.argument        := link.dataBase.U
  io.task.start.bits.descriptor.stack           := (link.dataBase + link.dataBytes).U
  io.task.start.bits.context.effectivePrivilege := 3.U
  when(io.task.start.fire)(state                := run)
  io.task.result.ready                          := state === run
  when(io.task.result.fire) {
    value := io.task.result.bits.value
    state := done
  }

  // The status of the task (bits 31:0) is the exit code.
  io.exit.valid := RegNext(io.task.result.fire, false.B)
  io.exit.bits  := value
}

/**
 * The virtual environment of a thread alone, standing in for its hart: the host that loads and launches its
 * program, the TSS, and the ideal memory behind its DMA. A DMA burst in the TSS window reaches the TSS, so a
 * workload checks the data its Balls produce; any other reaches the ideal memory (`axi`).
 */
@instantiable
class ThreadEnvironment(
  threadp:    ThreadParam,
  permission: PmpParams,
  simp:       SimParam,
  memoryBase: BigInt,
  factory:    ThreadFactory)
    extends Module {
  val link = threadp.link

  @public val io = IO(new Bundle {
    val axi  = new axi4.Port(link.dma)
    val exit = Output(Valid(UInt(64.W)))
  })

  val tlink: TLinkIO = factory.instantiate(threadp, permission, 0, simp, 0)
  val host    = Instantiate(new Host(link, permission, memoryBase))
  val tss     = Instantiate(new Store(StoreParams(link.tss, 2)))
  val dma     = Instantiate(new IdealDma(link.dma))
  val decoder = Instantiate(new axi4.Decoder(Seq((link.tss.base, BigInt(link.tss.bytes))), link.dma))
  val slave   = Instantiate(new StoreSlave(link.dma, link.tss))
  val memory  = Instantiate(new axi4.Interconnect(axi4.InterconnectParam(link.dma, 2, None)))

  host.io.task <> tlink.task
  tss.io.clients(0) <> tlink.tss
  tss.io.clients(1) <> slave.io.store
  tss.io.host <> host.io.tss
  tss.io.inUse  := true.B
  tss.io.cancel := VecInit(Seq.fill(2)(false.B))
  dma.io.dma <> tlink.dma
  decoder.io.in <> dma.io.axi
  slave.io.axi <> decoder.io.matched
  memory.io.in(0) <> host.io.axi
  memory.io.in(1) <> decoder.io.other
  io.axi <> memory.io.out
  io.exit       := host.io.exit
}

/** The Verilator top of a thread alone: its environment, the DPI DDR model and the exit of the simulation. */
class ThreadHarness(
  threadp:    ThreadParam,
  permission: PmpParams,
  simp:       SimParam,
  host:       buckyball.config.ThreadHost,
  factory:    ThreadFactory)
    extends Module {
  override def desiredName = "BBSimHarness"

  val memoryBase = BigInt(java.lang.Long.toUnsignedString(host.getMemoryBase))
  val clockDpi: Instance[BdbClkDPI] = Instantiate(new BdbClkDPI)
  clockDpi.io.clock := clock
  clockDpi.io.reset := reset.asBool

  val environment = Instantiate(new ThreadEnvironment(threadp, permission, simp, memoryBase, factory))

  val dram = Instantiate(new BBSimDRAM(
    threadp.link.dma,
    memoryBase,
    BigInt(java.lang.Long.toUnsignedString(host.getMemoryBytes)),
    simp.memory,
    0
  ))

  dram.io.clock := clock
  dram.io.reset := reset
  dram.io.axi <> environment.io.axi

  val scu = Instantiate(new SCUWriteDPI)
  scu.io.clock        := clock
  scu.io.reset        := reset.asBool
  scu.io.uart_hart_id := 0.U
  scu.io.uart_valid   := false.B
  scu.io.uart_data    := 0.U
  scu.io.exit_hart_id := 0.U
  scu.io.exit_valid   := environment.io.exit.valid
  scu.io.exit_code    := environment.io.exit.bits(31, 0)
}

/** Elaborates the Verilator top of the thread alone of design.pb's `thread`. */
object Verilator {

  def main(args: Array[String]): Unit = {
    require(args.length >= 6 && args(0) == "--config" && args(2) == "--target-class" && args(4) == "--target-dir")
    val design  = buckyball.config.Design.parseFrom(java.nio.file.Files.readAllBytes(java.nio.file.Paths.get(args(1))))
    require(design.hasThread && design.getThreadsCount == 1, s"${args(1)}: not the design.pb of a thread alone")
    val host    = design.getThread.getHost
    val threadp = ThreadParam.fromProto(design.getThreads(0), Elaborate.placement(host))
    val factory = Class.forName(threadp.rtlClass + "$").getField("MODULE$").get(null).asInstanceOf[ThreadFactory]
    ChiselStage.emitSystemVerilogFile(
      new ThreadHarness(
        threadp,
        PmpParams(host.getPmpAddressBits, host.getPmpEntries),
        SimParam.fromProto(design.getSims),
        host,
        factory
      ),
      args = Array("--target-dir", args(5), "--split-verilog"),
      firtoolOpts = args.drop(6)
    )
  }

}
