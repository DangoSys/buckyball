package framework.system.tile.tlink

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.ant.ControlRequest
import framework.memdomain.backend.shared.{SharedLeasePort, SharedPhysicalPort}

/** The controller stages one transfer; the start instruction returns after destination commit. */
@instantiable
class Control(p: T2TParams, localAccess: Boolean = false, leaseAccess: Boolean = true) extends Module {

  @public val io = IO(new Bundle {
    val request    = Flipped(Decoupled(new ControlRequest))
    val reply      = Decoupled(UInt(64.W))
    val tileId     = Input(UInt(p.tileBits.W))
    val command    = Decoupled(new T2TCommand(p))
    val completion = Flipped(Decoupled(Bool()))
    val idle       = Output(Bool())
    val local      = Option.when(localAccess)(new SharedPhysicalPort(128))
    val lease      = Option.when(leaseAccess)(new SharedLeasePort)
  })

  val idle :: running :: reading :: leasing :: responding :: Nil = Enum(5)
  val state                                                      = RegInit(idle)
  val fields                                                     = RegInit(0.U(4.W))
  val staged                                                     = Reg(Vec(4, UInt(64.W)))
  val answer                                                     = Reg(UInt(64.W))
  val request                                                    = io.request.bits
  val start                                                      = request.operation === 14.U
  val access                                                     = request.operation === 16.U || request.operation === 17.U
  val lease                                                      = request.operation === 18.U || request.operation === 19.U
  val upperHalf                                                  = Reg(Bool())
  val localWrite                                                 = Reg(Bool())
  val localReady                                                 = io.local.map(_.request.ready).getOrElse(false.B)
  val leaseReady                                                 = io.lease.map(_.request.ready).getOrElse(false.B)
  io.lease.foreach { port =>
    port.request.valid         := state === idle && io.request.valid && lease && !reset.asBool
    port.request.bits.endpoint := request.context
    port.request.bits.vbank    := request.field(31, 16)
    port.request.bits.group    := request.field(15, 0)
    port.request.bits.release  := request.operation === 19.U
    port.response.ready        := state === leasing && !reset.asBool
    when(port.response.fire) {
      answer := port.response.bits
      state  := responding
    }
  }
  io.local.foreach { port =>
    val address = Cat(request.context, request.field)
    port.request.valid        := state === idle && io.request.valid && access && !reset.asBool
    port.request.bits.address := address & ~15.U(64.W)
    port.request.bits.write   := request.operation === 17.U
    port.request.bits.data    := Cat(request.data, request.data)
    port.request.bits.mask    := Mux(address(3), "hff00".U, "h00ff".U)
    port.response.ready       := state === reading && !reset.asBool
    when(port.response.fire) {
      assert(!port.response.bits.error, "Main shared-storage access failed")
      answer := Mux(localWrite, 0.U, Mux(upperHalf, port.response.bits.data(127, 64), port.response.bits.data(63, 0)))
      state  := responding
    }
  }
  io.command.valid := state === idle && io.request.valid && start && !reset.asBool
  io.command.bits.sourceByteAddress := staged(0)
  io.command.bits.targetTile        := staged(1)
  io.command.bits.targetByteAddress := staged(2)
  io.command.bits.bytes             := staged(3)
  io.request.ready                  := state === idle && !reset.asBool &&
    Mux(lease, leaseReady, Mux(access, localReady, !start || io.command.ready))
  io.reply.valid                    := state === responding && !reset.asBool
  io.reply.bits                     := answer
  io.completion.ready               := state === running && !reset.asBool
  io.idle                           := state === idle
  when(io.request.valid && state === idle && !reset.asBool) {
    assert(
      (request.operation >= 13.U && request.operation <= 15.U) ||
        (localAccess.B && access) || (leaseAccess.B && lease),
      "Invalid TLink management operation"
    )
  }
  when(io.request.fire) {
    when(!access && !lease)(assert(request.context === 0.U, "TLink management has one tile-owned transfer context"))
    answer := 0.U
    state  := responding
    when(lease) {
      assert(request.data === 0.U, "TLink shared lease reserves rs2")
      state := leasing
    }
    when(access) {
      val address = Cat(request.context, request.field)
      assert(address(2, 0) === 0.U && (address +& 8.U) <= p.sharedBytes.U, "Invalid main shared-storage access")
      upperHalf  := address(3)
      localWrite := request.operation === 17.U
      state      := reading
    }
    switch(request.operation) {
      is(13.U) {
        assert(request.field < 4.U, "Invalid TLink transfer descriptor field")
        staged(request.field(1, 0)) := request.data
        fields                      := fields | UIntToOH(request.field, 4)
      }
      is(14.U) {
        assert(request.field === 0.U, "TLink start reserves the field selector")
        assert(fields.andR, "Incomplete TLink transfer descriptor")
        assert(
          staged(1) < p.tiles.U && staged(1) =/= io.tileId,
          "TLink transfer must target another tile"
        )
        assert(staged(3) > 0.U && (staged(3) >> 32) === 0.U, "Invalid TLink transfer size")
        fields := 0.U
        state  := running
      }
      is(15.U) {
        assert(request.field < 4.U, "Invalid TLink geometry query")
        answer := MuxLookup(request.field, 0.U)(Seq(
          0.U -> p.sharedBytes.U(64.W),
          1.U -> p.bankBytes.U(64.W),
          2.U -> io.tileId.pad(64),
          3.U -> p.tiles.U(64.W)
        ))
      }
    }
  }
  when(io.completion.fire) {
    assert(!io.completion.bits, "TLink transfer failed")
    state := responding
  }
  when(io.reply.fire)(state         := idle)
}
