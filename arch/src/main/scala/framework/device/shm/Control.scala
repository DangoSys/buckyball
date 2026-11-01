package framework.device.shm

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

/** The controller stages one transfer; the start instruction returns after destination commit. */
class ControlRequest extends Bundle {
  val operation = UInt(7.W)
  val context   = UInt(32.W)
  val field     = UInt(32.W)
  val data      = UInt(64.W)
}

@instantiable
class Control(transferParam: Params) extends Module {
  val p = transferParam

  @public
  val io = IO(new Bundle {
    val request        = Flipped(Decoupled(new ControlRequest))
    val reply          = Decoupled(UInt(64.W))
    val coreId         = Input(UInt(p.coreBits.W))
    val routeTarget    = Output(UInt(64.W))
    val routeAvailable = Input(Bool())
    val command        = Decoupled(new C2CCommand(transferParam))
    val completion     = Flipped(Decoupled(Bool()))
    val idle           = Output(Bool())
    val local          = new StorageAccess(transferParam.axi.dataBits)
  })

  val idle :: running :: reading :: responding :: Nil = Enum(4)
  val state                                           = RegInit(idle)
  val fields                                          = RegInit(0.U(4.W))
  val staged                                          = Reg(Vec(4, UInt(64.W)))
  val answer                                          = Reg(UInt(64.W))
  val request                                         = io.request.bits
  val start                                           = request.operation === 14.U
  val routeQuery                                      = request.operation === 15.U && request.field === 3.U
  io.routeTarget := Mux(routeQuery, request.data, staged(1))
  val access     = request.operation === 16.U || request.operation === 17.U
  val upperHalf  = Reg(Bool())
  val localWrite = Reg(Bool())
  val localReady = io.local.request.ready
  val local      = io.local
  val address    = Cat(request.context, request.field)
  local.request.valid               := state === idle && io.request.valid && access && !reset.asBool
  local.request.bits.address        := address & ~15.U(64.W)
  local.request.bits.write          := request.operation === 17.U
  local.request.bits.data           := Cat(request.data, request.data)
  local.request.bits.mask           := Mux(address(3), "hff00".U, "h00ff".U)
  local.response.ready              := state === reading && !reset.asBool
  when(local.response.fire) {
    assert(!local.response.bits.error, "Main shared-storage access failed")
    answer := Mux(localWrite, 0.U, Mux(upperHalf, local.response.bits.data(127, 64), local.response.bits.data(63, 0)))
    state  := responding
  }
  io.command.valid                  := state === idle && io.request.valid && start && io.routeAvailable && !reset.asBool
  io.command.bits.sourceByteAddress := staged(0)
  io.command.bits.targetCore        := staged(1)
  io.command.bits.targetByteAddress := staged(2)
  io.command.bits.bytes             := staged(3)
  io.request.ready                  := state === idle && !reset.asBool &&
    Mux(access, localReady, !start || !io.routeAvailable || io.command.ready)
  io.reply.valid                    := state === responding && !reset.asBool
  io.reply.bits                     := answer
  io.completion.ready               := state === running && !reset.asBool
  io.idle                           := state === idle
  when(io.request.valid && state === idle && !reset.asBool) {
    assert(
      request.operation >= 13.U && request.operation <= 17.U,
      "Invalid CLink management operation"
    )
  }
  when(io.request.fire) {
    when(!access)(assert(request.context === 0.U, "CLink management has one core-owned transfer context"))
    answer := 0.U
    state  := responding
    when(access) {
      val address = Cat(request.context, request.field)
      assert(address(2, 0) === 0.U && (address +& 8.U) <= p.capacity.U, "Invalid main shared-storage access")
      upperHalf  := address(3)
      localWrite := request.operation === 17.U
      state      := reading
    }
    switch(request.operation) {
      is(13.U) {
        assert(request.field < 4.U, "Invalid CLink transfer descriptor field")
        staged(request.field(1, 0)) := request.data
        fields                      := fields | UIntToOH(request.field, 4)
      }
      is(14.U) {
        assert(request.field === 0.U, "CLink start reserves the field selector")
        assert(fields.andR, "Incomplete CLink transfer descriptor")
        assert(staged(3) > 0.U && (staged(3) >> 32) === 0.U, "Invalid CLink transfer size")
        fields                        := 0.U
        answer                        := !io.routeAvailable
        when(io.routeAvailable)(state := running)
      }
      is(15.U) {
        assert(request.field <= 3.U, "Invalid CLink query")
        answer := MuxLookup(request.field, 0.U)(Seq(
          0.U -> p.capacity.U(64.W),
          1.U -> io.coreId.pad(64),
          2.U -> p.cores.U(64.W),
          3.U -> Cat(0.U(63.W), io.routeAvailable)
        ))
      }
    }
  }
  when(io.completion.fire) {
    answer := io.completion.bits.asUInt
    state  := responding
  }
  when(io.reply.fire)(state         := idle)
}
