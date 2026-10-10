package framework.ant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import memcore.memory.{spm, tss}

/**
 * Tile execution service. The caller supplies the uncached management transport and
 * task-bound NPU/DMA owner; neither CPU cache nor address translation lives here.
 */
@instantiable
class Service(p: Params, contexts: Int) extends Module {

  @public val io = IO(new Bundle {
    val request    = Flipped(Decoupled(new ControlRequest))
    val reply      = Decoupled(UInt(64.W))
    val signatures = Input(Vec(contexts, UInt(64.W)))
    val online     = Input(Vec(contexts, Bool()))
    val locals     = Vec(contexts, Flipped(new LocalPort(p)))
    val launched   = Output(Vec(contexts, Valid(new Start(p))))
    val retired    = Output(Vec(contexts, Valid(new Retire)))
    val idle       = Output(Bool())
  })

  val control     = Instantiate(new Control(p, contexts))
  val sharedStore = Instantiate(new tss.Store(tss.Params(p.shared, contexts)))
  val running     = VecInit(io.locals.map(_.running))
  val storageBusy = io.locals.map(_.storageBusy).reduce(_ || _) ||
    sharedStore.io.busy.asUInt.orR || sharedStore.io.hostBusy
  control.io.signatures  := io.signatures
  control.io.online      := io.online
  control.io.storageBusy := storageBusy
  sharedStore.io.inUse   := control.io.inUse || running.asUInt.orR
  sharedStore.io.cancel  := control.io.cancel
  for (i <- 0 until contexts) {
    io.locals(i).inUse          := control.io.inUse
    io.locals(i).start <> control.io.start(i)
    control.io.result(i) <> io.locals(i).result
    io.locals(i).cancel         := control.io.cancel(i)
    io.locals(i).sharedBusy     := sharedStore.io.busy(i)
    io.locals(i).sharedHostBusy := sharedStore.io.hostBusy
    sharedStore.io.clients(i) <> io.locals(i).shared
    io.retired(i)               := io.locals(i).retired
  }
  val idle :: controlReply :: loading :: loadReply :: Nil = Enum(4)
  val state      = RegInit(idle)
  val request    = io.request.bits
  val isLoad     = request.operation >= 7.U
  val space      = (request.operation - 7.U) >> 1
  val write      = !request.operation(0)
  val shared     = space === 2.U
  val localIndex = request.context(math.max(1, log2Ceil(contexts)) - 1, 0)
  val offset     = request.field
  val bytes      = MuxLookup(space, 0.U)(Seq(0.U -> p.codeBytes.U, 1.U -> p.data.bytes.U, 2.U -> p.shared.bytes.U))
  val localBase  = MuxLookup(space, 0.U(64.W))(Seq(0.U -> 0.U, 1.U -> p.data.base.U, 2.U -> p.shared.base.U))
  val valid      = request.operation <= 12.U && Mux(shared, request.context === 0.U, request.context < contexts.U) &&
    offset(2, 0) === 0.U && (offset +& 8.U) <= bytes
  val permitted  =
    Mux(shared, !control.io.inUse, !running(localIndex) && (!write || control.io.available(localIndex)))
  val ports      = io.locals.map(_.code).toSeq ++ io.locals.map(_.data).toSeq :+ sharedStore.io.host
  val selected   = Mux(shared, (2 * contexts).U, Mux(space === 0.U, request.context, request.context +& contexts.U))
  val owner      = Reg(UInt(log2Ceil(ports.size).W))
  val value      = Reg(UInt(64.W))
  for ((port, i) <- ports.zipWithIndex) {
    port.request.valid        := state === idle && io.request.valid && isLoad && valid && permitted && selected === i.U && !reset.asBool
    port.request.bits         := 0.U.asTypeOf(port.request.bits)
    port.request.bits.address := localBase + offset
    port.request.bits.size    := 3.U
    port.request.bits.write   := write
    port.request.bits.data    := request.data
    port.request.bits.mask    := Mux(write, 255.U, 0.U)
    port.response.ready       := state === loading && owner === i.U && !reset.asBool
  }
  when(state === idle && io.request.valid && isLoad && !reset.asBool) {
    assert(valid, "Invalid Ant local loader address or operation")
    assert(permitted, "Ant local storage loader conflicts with execution ownership")
  }
  control.io.request.valid := state === idle && io.request.valid && !isLoad && !reset.asBool
  control.io.request.bits   := request
  io.request.ready          := state === idle && !reset.asBool &&
    Mux(isLoad, valid && permitted && VecInit(ports.map(_.request.ready))(selected), control.io.request.ready)
  when(io.request.fire) {
    state := Mux(isLoad, loading, controlReply)
    owner := selected
  }
  val response = VecInit(ports.map(_.response.bits))(owner)
  when(state === loading && VecInit(ports.map(_.response.valid))(owner)) {
    assert(!response.error, "Ant local loader storage access failed")
    value := response.data(63, 0)
    state := loadReply
  }
  io.reply.valid            := (state === loadReply || (state === controlReply && control.io.response.valid)) && !reset.asBool
  io.reply.bits             := Mux(state === loadReply, value, control.io.response.bits)
  control.io.response.ready := state === controlReply && io.reply.ready && !reset.asBool
  when(io.reply.fire)(state := idle)
  io.idle                   := state === idle && !running.asUInt.orR && !storageBusy
  for (i <- 0 until contexts) {
    io.launched(i).valid := control.io.start(i).fire
    io.launched(i).bits  := control.io.start(i).bits
  }
}
