package framework.tlp.memory

import framework.hart.memory.{_}
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.tlp.shm.{MeshChannel, MeshSharedMemParams}
import framework.generic.bus.axi4
import framework.generic.dma.Dma

class Command extends Bundle {
  val operation = UInt(7.W)
  val context   = UInt(32.W)
  val field     = UInt(32.W)
  val data      = UInt(64.W)
}

/** A compiled data edge names tasks and ordinary addresses, never thread banks. */
class Flow(p: MeshSharedMemParams) extends Bundle {
  val source        = UInt(p.threadBits.W)
  val sourceTask    = UInt(64.W)
  val sourceAddress = UInt(64.W)
  val sourceEnd     = UInt(64.W)
  val target        = UInt(p.threadBits.W)
  val targetTask    = UInt(64.W)
  val targetAddress = UInt(64.W)
  val targetEnd     = UInt(64.W)
  val bytes         = UInt(32.W)
  val bank          = UInt(p.bankBits.W)
  val live          = Bool()
  val published     = Bool()
}

/** Compiled dataflow bindings and routing; storage is connected by the hart. */
@instantiable
class Dataflow(p: MeshSharedMemParams, bus: axi4.InterconnectParam) extends Module {
  require(p.threads.size == bus.masters && p.threads.forall(_.ports == 2))
  require(p.externalChannels == 0 && p.dataBits == bus.bus.dataBits && p.tagBits >= 2)

  @public val io = IO(new Bundle {
    val command  = Flipped(Decoupled(new Command))
    val reply    = Decoupled(UInt(64.W))
    val task     = Input(Vec(p.threads.size, Valid(UInt(64.W))))
    val threads  = Vec(p.threads.size, Flipped(new Dma(bus.bus)))
    val external = Vec(p.threads.size, new TranslatedDma(bus.bus))
    val storage  = Flipped(Vec(p.totalChannels, new MeshChannel(p)))
    val busy     = Output(Bool())
    val stalled  = Output(Vec(p.threads.size, Bool()))
  })

  val command = io.command.bits
  val index   = command.field(p.bankBits - 1, 0)
  val flows   = RegInit(VecInit(Seq.fill(p.bankCount)(0.U.asTypeOf(new Flow(p)))))
  val current = RegInit(VecInit(Seq.fill(p.threads.size)(0.U(64.W))))

  val endpoints = p.threads.indices.map { thread =>
    val endpoint = Instantiate(new DmaRoute(p, bus.bus, thread))
    endpoint.io.thread <> io.threads(thread)
    io.external(thread).read <> endpoint.io.external.read
    io.external(thread).write <> endpoint.io.external.write
    io.external(thread).writeData <> endpoint.io.external.writeData
    endpoint.io.external.readData <> io.external(thread).readData
    endpoint.io.external.completed <> io.external(thread).completed
    // The owner of the thread resolves a page fault that stalls its DDR burst.
    io.external(thread).resolve.valid           := io.command.fire && command.operation === 25.U && command.context === thread.U
    io.external(thread).resolve.bits            := command.data(0)
    endpoint.io.flows                           := flows
    endpoint.io.task                            := current(thread)
    endpoint.io.read <> io.storage(2 * thread)
    endpoint.io.write <> io.storage(2 * thread + 1)
    when(io.task(thread).valid)(current(thread) := io.task(thread).bits)
    endpoint
  }

  val bankBytes = p.entriesPerBank * p.maskBits
  val bankShift = log2Ceil(bankBytes)

  val occupied = VecInit((0 until p.bankCount).map(bank =>
    flows.map(f =>
      f.live && bank.U >= f.bank && bank.U < f.bank +& ((f.bytes +& (bankBytes - 1).U) >> bankShift)
    ).reduce(_ || _)
  ))

  val busy = VecInit((0 until p.bankCount).map(bank => endpoints.map(_.io.busy(bank)).reduce(_ || _)))
  io.busy := busy.asUInt.orR

  val staged  = Reg(Vec(7, UInt(64.W)))
  val fields  = RegInit(0.U(7.W))
  val pending = RegInit(false.B)
  val result  = Reg(UInt(64.W))
  val free    = VecInit(flows.map(f => !f.live))
  val slot    = PriorityEncoder(free)

  val existing = VecInit(flows.map(f =>
    f.live && f.source === staged(0) &&
      f.sourceTask === staged(1) && f.sourceAddress === staged(2) && f.bytes === staged(6)
  ))

  val fault = VecInit(io.external.map(_.fault))
  io.stalled := VecInit(fault.map(_.valid))
  val owner = command.context(math.max(1, log2Ceil(p.threads.size)) - 1, 0)

  val bankCountBits = log2Ceil(p.bankCount + 1)
  val required      = ((staged(6) +& (bankBytes - 1).U) >> bankShift)(bankCountBits - 1, 0)

  val fits =
    VecInit(
      (0 until p.bankCount).map(first =>
        first.U +& required <= p.bankCount.U &&
          !(0 until p.bankCount).map(bank =>
            occupied(bank) && bank.U >= first.U && bank.U < first.U +& required
          ).reduce(_ || _)
      )
    )

  val storage = Mux(existing.asUInt.orR, flows(PriorityEncoder(existing)).bank, PriorityEncoder(fits))
  io.command.ready            := !pending && !reset.asBool
  io.reply.valid              := pending && !reset.asBool
  io.reply.bits               := result
  when(io.reply.fire)(pending := false.B)
  when(io.command.fire) {
    assert(
      Mux(
        command.operation >= 24.U,
        command.operation === 24.U || command.operation === 25.U && fault(owner).valid,
        command.context === 0.U && command.operation >= 20.U
      ) && command.context < p.threads.size.U,
      "Invalid hart dataflow command"
    )
    pending := true.B
    result  := 0.U
    switch(command.operation) {
      is(20.U) {
        assert(command.field < 7.U, "Invalid dataflow descriptor field")
        staged(command.field(2, 0)) := command.data
        fields                      := fields | UIntToOH(command.field, 7)
      }
      is(21.U) {
        assert(fields.andR && free.asUInt.orR, "Incomplete dataflow descriptor or no free edge")
        assert(staged(0) < p.threads.size.U && staged(3) < p.threads.size.U, "Invalid dataflow thread")
        assert(
          staged(6) > 0.U && staged(6) <= (bankBytes * p.bankCount).U &&
            !(staged(2) | staged(5) | staged(6))(log2Ceil(p.maskBits) - 1, 0).orR,
          "Hart dataflow must contain aligned complete bank rows"
        )
        assert(existing.asUInt.orR || fits.asUInt.orR, "Hart dataflow storage exhausted")
        assert(!(staged(2) +& staged(6))(64) && !(staged(5) +& staged(6))(64), "Dataflow range overflow")
        for (flow <- flows) {
          when(flow.live && flow.target === staged(3) && flow.targetTask === staged(4)) {
            assert(
              staged(5) >= flow.targetAddress +& flow.bytes || flow.targetAddress >= staged(5) +& staged(6),
              "Overlapping dataflow inputs in one task"
            )
          }
        }
        flows(slot).source        := staged(0)
        flows(slot).sourceTask    := staged(1)
        flows(slot).sourceAddress := staged(2)
        flows(slot).sourceEnd     := staged(2) + staged(6)
        flows(slot).target        := staged(3)
        flows(slot).targetTask    := staged(4)
        flows(slot).targetAddress := staged(5)
        flows(slot).targetEnd     := staged(5) + staged(6)
        flows(slot).bytes         := staged(6)
        flows(slot).bank          := storage
        flows(slot).live          := true.B
        flows(slot).published     := existing.asUInt.orR && flows(PriorityEncoder(existing)).published
        fields                    := 0.U
        result                    := slot
      }
      is(22.U) {
        assert(command.field < p.bankCount.U && flows(index).live, "Unknown dataflow edge")
        assert(
          !(0 until p.bankCount).map(bank =>
            busy(bank) && bank.U >= flows(index).bank &&
              bank.U < flows(index).bank +& ((flows(index).bytes +& (bankBytes - 1).U) >> bankShift)
          ).reduce(_ || _),
          "Dataflow released with memory operations in flight"
        )
        flows(index).live := false.B
      }
      // The page fault of thread `context`: bit 63 a burst is stalled, bit 62 it is a store,
      // bits 61:0 its virtual address.
      is(24.U) {
        result := Cat(fault(owner).valid, fault(owner).bits.write, fault(owner).bits.address(61, 0))
      }
      is(23.U) {
        assert(command.field < p.bankCount.U && flows(index).live, "Unknown producer edge")
        assert(
          !(0 until p.bankCount).map(bank =>
            busy(bank) && bank.U >= flows(index).bank &&
              bank.U < flows(index).bank +& ((flows(index).bytes +& (bankBytes - 1).U) >> bankShift)
          ).reduce(_ || _),
          "Dataflow published before producer writes complete"
        )
        for (flow <- flows) {
          when(flow.live && flow.bank === flows(index).bank)(flow.published := true.B)
        }
      }
    }
  }
}
