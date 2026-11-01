package framework.core.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.ip.bus.axi4
import framework.thread.tlink.Dma

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
  val target        = UInt(p.threadBits.W)
  val targetTask    = UInt(64.W)
  val targetAddress = UInt(64.W)
  val bytes         = UInt(32.W)
  val bank          = UInt(p.bankBits.W)
  val live          = Bool()
  val published     = Bool()
}

/** Core-owned buffering behind ordinary thread DMA. Loom publishes completed producers. */
@instantiable
class Exchange(p: MeshSharedMemParams, bus: axi4.InterconnectParam) extends Module {
  require(p.threads.size == bus.masters && p.threads.forall(_.ports == 2))
  require(p.externalChannels == 0 && p.dataBits == bus.bus.dataBits && p.tagBits >= 2)

  @public val io = IO(new Bundle {
    val command = Flipped(Decoupled(new Command))
    val reply   = Decoupled(UInt(64.W))
    val task    = Input(Vec(p.threads.size, Valid(UInt(64.W))))
    val threads = Vec(p.threads.size, Flipped(new Dma(bus.bus)))
    val ddr     = new axi4.Port(bus.bus)
    val busy    = Output(Bool())
  })

  val memory = Instantiate(new MeshStorage(p, transfers = false))
  memory.io.transferCommand.valid      := false.B
  memory.io.transferCommand.bits       := 0.U.asTypeOf(memory.io.transferCommand.bits)
  memory.io.transferCompletion.ready   := true.B
  memory.io.localBanks.foreach { local =>
    local.request.ready  := false.B
    local.response.valid := false.B
    local.response.bits  := 0.U.asTypeOf(local.response.bits)
  }
  memory.io.bankWrites.foreach(_.ready := true.B)
  val ddr     = Instantiate(new axi4.Interconnect(bus))
  io.ddr <> ddr.io.out
  val flows   = RegInit(VecInit(Seq.fill(p.bankCount)(0.U.asTypeOf(new Flow(p)))))
  val current = RegInit(VecInit(Seq.fill(p.threads.size)(0.U(64.W))))

  val endpoints = p.threads.indices.map { thread =>
    val endpoint = Instantiate(new Endpoint(p, bus.bus, thread))
    endpoint.io.thread <> io.threads(thread)
    endpoint.io.ddr <> ddr.io.in(thread)
    endpoint.io.flows                           := flows
    endpoint.io.task                            := current(thread)
    endpoint.io.read <> memory.io.channels(2 * thread)
    endpoint.io.write <> memory.io.channels(2 * thread + 1)
    when(io.task(thread).valid)(current(thread) := io.task(thread).bits)
    endpoint
  }

  val occupied = VecInit((0 until p.bankCount).map(bank => flows.map(f => f.live && f.bank === bank.U).reduce(_ || _)))
  val busy     = VecInit((0 until p.bankCount).map(bank => endpoints.map(_.io.busy(bank)).reduce(_ || _)))
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

  val storage = Mux(existing.asUInt.orR, flows(PriorityEncoder(existing)).bank, PriorityEncoder(~occupied.asUInt))
  val command = io.command.bits
  val index   = command.field(p.bankBits - 1, 0)
  io.command.ready            := !pending && !reset.asBool
  io.reply.valid              := pending && !reset.asBool
  io.reply.bits               := result
  when(io.reply.fire)(pending := false.B)
  when(io.command.fire) {
    assert(
      command.context === 0.U && command.operation >= 20.U && command.operation <= 23.U,
      "Invalid core dataflow command"
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
          staged(6) > 0.U && staged(6) <= (p.entriesPerBank * p.maskBits).U &&
            !(staged(2) | staged(5) | staged(6))(log2Ceil(p.maskBits) - 1, 0).orR,
          "Core dataflow must contain aligned complete bank rows"
        )
        assert(existing.asUInt.orR || !occupied.asUInt.andR, "Core dataflow storage exhausted")
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
        flows(slot).target        := staged(3)
        flows(slot).targetTask    := staged(4)
        flows(slot).targetAddress := staged(5)
        flows(slot).bytes         := staged(6)
        flows(slot).bank          := storage
        flows(slot).live          := true.B
        flows(slot).published     := existing.asUInt.orR && flows(PriorityEncoder(existing)).published
        fields                    := 0.U
        result                    := slot
      }
      is(22.U) {
        assert(command.field < p.bankCount.U && flows(index).live, "Unknown dataflow edge")
        assert(!busy(flows(index).bank), "Dataflow released with memory operations in flight")
        flows(index).live := false.B
      }
      is(23.U) {
        assert(command.field < p.bankCount.U && flows(index).live, "Unknown producer edge")
        assert(!busy(flows(index).bank), "Dataflow published before producer writes complete")
        for (flow <- flows) {
          when(flow.live && flow.bank === flows(index).bank)(flow.published := true.B)
        }
      }
    }
  }
}
