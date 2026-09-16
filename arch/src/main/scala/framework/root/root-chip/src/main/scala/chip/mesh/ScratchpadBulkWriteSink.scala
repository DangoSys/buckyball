package hier.chip.mesh

import chisel3._
import chisel3.util._
import memcore.bus.axi.AxiSBeat
import memcore.memory.bank.{RootScratchpad, ScratchpadParams}

/** NoC extension that binds VC5/VC4/VC6 DMA traffic to a local scratchpad. */
class ScratchpadBulkWriteSink(p: ScratchpadParams, nodeIdBits: Int) extends Module {

  val io = IO(new Bundle {
    val descriptor = Flipped(Decoupled(new BulkDescriptor(nodeIdBits)))
    val bulkIn     = Flipped(Decoupled(new AxiSBeat(p.dataBits)))
    val completion = Decoupled(new BulkCompletion(nodeIdBits))
  })

  val scratchpad                                    = Module(new RootScratchpad(p))
  val idle :: issue :: streaming :: complete :: Nil = Enum(4)
  val state                                         = RegInit(idle)
  val descriptor                                    = Reg(new BulkDescriptor(nodeIdBits))
  val beats                                         = descriptor.bytes / p.bytes.U
  io.descriptor.ready                    := state === idle
  when(io.descriptor.fire) {
    assert(
      io.descriptor.bits.write && io.descriptor.bits.bytes =/= 0.U &&
        io.descriptor.bits.bytes % p.bytes.U === 0.U,
      "Scratchpad bulk sink accepts aligned nonempty writes only"
    )
    descriptor                  := io.descriptor.bits
    state                       := issue
  }
  scratchpad.io.command.valid            := state === issue
  scratchpad.io.command.bits.write       := true.B
  scratchpad.io.command.bits.addr        := descriptor.scratchpadAddr(p.addressBits - 1, 0)
  scratchpad.io.command.bits.beats       := beats
  when(scratchpad.io.command.fire)(state := streaming)
  scratchpad.io.write.valid              := state === streaming && io.bulkIn.valid
  scratchpad.io.write.bits               := io.bulkIn.bits
  io.bulkIn.ready                        := state === streaming && scratchpad.io.write.ready
  scratchpad.io.read.ready               := true.B
  scratchpad.io.done.ready               := state === streaming
  when(scratchpad.io.done.fire)(state    := complete)
  io.completion.valid                    := state === complete
  io.completion.bits.targetNode          := descriptor.sourceNode
  io.completion.bits.txnId               := descriptor.txnId
  io.completion.bits.ready               := false.B
  io.completion.bits.error               := scratchpad.io.done.bits
  when(io.completion.fire)(state         := idle)
}
