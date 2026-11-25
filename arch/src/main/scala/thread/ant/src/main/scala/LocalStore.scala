package framework.thread.ant

import framework.thread.ant.configs.AntParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.generic.memory.spm.{Memory, Params, Port}

/**
 * A context-private local store. Loader and execution access it in separate ownership phases.
 * Cancellation drains an accepted local access and discards its response before reuse.
 */
@instantiable
class LocalStore(val b: AntParam, index: Int) extends Module {
  val local = b

  val (p, localReadOnly) = index match {
    case 0 => (Params(0, local.codeBytes, local.data.dataBits), true)
    case 1 => (local.data, false)
  }

  @public
  val io = IO(new Bundle {
    val running = Input(Bool())
    val cancel  = Input(Bool())

    val host = Flipped(new Port(p))

    val local = Flipped(new Port(p))

    val busy = Output(Bool())
  })

  val bankBytes = if (index == 0) math.min(p.bytes, 16 * 1024) else p.bytes

  val memories = (0 until p.bytes / bankBytes).map { bank =>
    Instantiate(new Memory(Params(p.base + bank * bankBytes, bankBytes, p.dataBits)))
  }

  val bankBits     = math.max(1, log2Ceil(memories.size))
  val returnedBank = Reg(UInt(bankBits.W))
  val request      = Mux(io.running, io.local.request.bits, io.host.request.bits)

  val selectedBank =
    if (memories.size == 1) 0.U
    else
      ((request.address - p.base.U) >> log2Ceil(bankBytes))(bankBits - 1, 0)

  val requestReady  = VecInit(memories.map(_.io.port.request.ready))(selectedBank)
  val responseValid = VecInit(memories.map(_.io.port.response.valid))(returnedBank)
  val responseBits  = VecInit(memories.map(_.io.port.response.bits))(returnedBank)

  val busy      = RegInit(false.B)
  val owner     = Reg(Bool())
  val cancelled = RegInit(false.B)
  val discard   = owner && (cancelled || io.cancel)
  for ((memory, bank) <- memories.zipWithIndex) {
    memory.io.port.request.valid  := !busy && !reset.asBool && selectedBank === bank.U &&
      Mux(io.running, io.local.request.valid && !io.cancel, io.host.request.valid)
    memory.io.port.request.bits   := request
    memory.io.readOnly            := io.running && localReadOnly.B
    memory.io.port.response.ready := busy && returnedBank === bank.U &&
      (discard || Mux(owner, io.local.response.ready, io.host.response.ready))
  }
  io.local.request.ready := !busy && io.running && !io.cancel && requestReady
  io.host.request.ready                      := !busy && !io.running && requestReady
  when(VecInit(memories.map(_.io.port.request.fire)).asUInt.orR) {
    busy         := true.B
    owner        := io.running
    cancelled    := false.B
    returnedBank := selectedBank
  }
  when(busy && owner && io.cancel)(cancelled := true.B)
  io.local.response.valid                    := busy && owner && !discard && responseValid
  io.host.response.valid                     := busy && !owner && responseValid
  io.local.response.bits                     := responseBits
  io.host.response.bits                      := responseBits
  when(busy && responseValid &&
    (discard || Mux(owner, io.local.response.ready, io.host.response.ready))) {
    busy      := false.B
    cancelled := false.B
  }
  io.busy                                    := busy
}
