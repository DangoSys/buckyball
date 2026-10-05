package framework.memdomain.frontend.mem.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import memcore.memory.preflight.{Error, MapQuery, MapResult, Params}
import memcore.bus.axi4

@instantiable
class ReadDma(b: GlobalConfig, prepared: Params, axiParams: axi4.Params) extends Module {
  require(b.memDomain.bankWidth == 128 && b.memDomain.dma_buswidth == 128 && axiParams.dataBits == 128)
  require(prepared.beatBytes == 16 && prepared.bus.addressBits == axiParams.addressBits)

  @public
  val io = IO(new Bundle {
    val req           = Flipped(Decoupled(new BBReadRequest))
    val resp          = Decoupled(new BBReadResponse(128))
    val parentId      = Input(UInt(prepared.idBits.W))
    val decisionValid = Input(Bool())
    val decisionFault = Input(new DmaStatus)
    val busy          = Output(Bool())
    val query         = Output(new MapQuery(prepared))
    val mapping       = Input(new MapResult(prepared))
    val axi           = new axi4.Port(axiParams)
  })

  val idle :: shape :: plan :: query :: offer :: receive :: respond :: drain :: failed :: Nil = Enum(9)
  val state                                                                                   = RegInit(idle)
  val command                                                                                 = Reg(new BBReadRequest)
  val parent                                                                                  = Reg(UInt(prepared.idBits.W))
  val total                                                                                   = Reg(UInt(17.W))
  val progress                                                                                = RegInit(0.U(17.W))
  val group                                                                                   = RegInit(0.U(6.W))
  val column                                                                                  = RegInit(0.U(4.W))
  val address                                                                                 = Reg(UInt(64.W))
  val queryVA                                                                                 = Reg(UInt(64.W))
  val physical                                                                                = Reg(UInt(axiParams.addressBits.W))
  val candidate                                                                               = Reg(UInt(9.W))
  val remaining                                                                               = Reg(UInt(9.W))
  val second                                                                                  = RegInit(false.B)
  val firstPart                                                                               = Reg(UInt(128.W))
  val result                                                                                  = Reg(UInt(128.W))
  val fault                                                                                   = RegInit(0.U.asTypeOf(new DmaStatus))

  io.req.ready             := state === idle && io.decisionValid
  io.busy                  := state =/= idle
  io.resp.valid            := state === respond || state === failed
  io.resp.bits.data        := Mux(state === failed, 0.U, result)
  io.resp.bits.last        := state === failed || progress +& 1.U === total
  io.resp.bits.addrcounter := progress(15, 0)
  io.resp.bits.fault       := Mux(state === failed, fault, 0.U.asTypeOf(new DmaStatus))
  io.query                 := 0.U.asTypeOf(new MapQuery(prepared))
  io.query.valid           := state === query
  io.query.id              := parent
  io.query.va              := queryVA
  io.query.bytes           := candidate << 4
  io.query.write           := false.B
  io.axi.ar.valid          := state === offer
  io.axi.ar.bits           := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.ar.bits.addr      := physical
  io.axi.ar.bits.len       := candidate - 1.U
  io.axi.ar.bits.size      := 4.U
  io.axi.ar.bits.burst     := 1.U
  io.axi.r.ready           := state === receive || state === drain
  io.axi.aw.valid          := false.B
  io.axi.aw.bits           := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.w.valid           := false.B
  io.axi.w.bits            := 0.U.asTypeOf(new axi4.WriteData(axiParams))
  io.axi.b.ready           := false.B

  def fail(error: UInt, at: UInt): Unit = {
    fault.error   := error
    fault.address := at
    state         := failed
  }

  val count = (io.req.bits.len.pad(33) + 15.U) >> 4
  when(io.req.fire) {
    command := io.req.bits; parent         := io.parentId
    address := io.req.bits.vaddr; progress := 0.U; group := 0.U; column := 0.U; second := false.B
    fault   := 0.U.asTypeOf(new DmaStatus)
    val bad = io.req.bits.len === 0.U || count > 65536.U ||
      io.req.bits.groups === 0.U || io.req.bits.stride === 0.U ||
      (io.req.bits.is_2d && (io.req.bits.pixel_bytes === 0.U || io.req.bits.tile_width === 0.U ||
        io.req.bits.source_width < io.req.bits.tile_width))
    when(io.decisionFault.error =/= DmaError.None.U) {
      fail(io.decisionFault.error, io.decisionFault.address)
    }.elsewhen(bad)(fail(DmaError.Shape.U, io.req.bits.vaddr)).otherwise {
      total := count(16, 0); state := shape
    }
  }

  val divisor      = Mux(command.is_2d, command.tile_width.pad(6), command.groups)
  val rowStride    = Mux(command.is_2d, command.source_width * command.pixel_bytes, command.groups * command.stride * 16.U)
  val columnStride = Mux(command.is_2d, command.pixel_bytes, 16.U)
  val lastIndex    = total - 1.U
  val lastByte     = command.vaddr.pad(66) +& ((lastIndex / divisor) * rowStride) +&
    ((lastIndex % divisor) * columnStride) +& 15.U
  when(state === shape) {
    when((lastByte >> 64).orR)(fail(DmaError.Shape.U, command.vaddr)).otherwise(state := plan)
  }

  val contiguous = Mux(
    command.is_2d,
    Mux(
      command.pixel_bytes =/= 16.U,
      1.U,
      Mux(command.source_width === command.tile_width, total - progress, command.tile_width - column)
    ),
    Mux(command.stride === 1.U, total - progress, command.groups - group)
  )

  val pageBeats    = (4096.U(13.W) - address(11, 0)) >> 4
  val commandBeats = Mux(contiguous < (total - progress), contiguous, total - progress)
  val limited      = Mux(commandBeats < 256.U, commandBeats, 256.U)
  when(state === plan) {
    queryVA   := address & ~15.U(64.W)
    candidate := Mux(address(3, 0) =/= 0.U, 1.U, Mux(limited < pageBeats, limited, pageBeats))
    state     := query
  }
  when(state === query) {
    when(!io.mapping.hit || io.mapping.error =/= Error.Ok.U) {
      val error = MuxLookup(io.mapping.error, DmaError.AccessFault.U)(Seq(
        Error.Shape.U     -> DmaError.Shape.U,
        Error.Overflow.U  -> DmaError.Shape.U,
        Error.PageFault.U -> DmaError.PageFault.U,
        Error.Capacity.U  -> DmaError.Capacity.U,
        Error.Context.U   -> DmaError.Context.U
      ))
      fail(error, queryVA)
    }.elsewhen(io.mapping.pa(3, 0) =/= 0.U || (io.mapping.pa(11, 0) +& (candidate << 4)) > 4096.U) {
      fail(DmaError.AccessFault.U, queryVA)
    }.otherwise { physical := io.mapping.pa; state := offer }
  }
  when(io.axi.ar.fire) { remaining := candidate; state := receive }
  when(io.axi.r.fire) {
    when(state === drain) {
      when(io.axi.r.bits.last)(state := failed)
    }.otherwise {
      val badId       = io.axi.r.bits.id =/= 0.U
      val badLast     = io.axi.r.bits.last =/= (remaining === 1.U)
      val badResponse = io.axi.r.bits.resp =/= 0.U
      assert(!badLast, "ReadDma AXI RLAST does not match ARLEN")
      when(badId || badLast || badResponse) {
        fault.error   := Mux(badId || badLast || io.axi.r.bits.resp === 1.U, DmaError.Protocol.U, DmaError.AccessFault.U)
        fault.address := physical
        state         := Mux(badLast || io.axi.r.bits.last, failed, drain)
      }.otherwise {
        remaining := remaining - 1.U
        physical  := physical + 16.U
        when(address(3, 0) =/= 0.U && !second) {
          firstPart := (io.axi.r.bits.data >> (address(3, 0) * 8.U))(127, 0)
          second    := true.B; queryVA := queryVA + 16.U; candidate := 1.U; state := query
        }.otherwise {
          result := Mux(
            second,
            firstPart | (io.axi.r.bits.data << ((16.U - address(3, 0)) * 8.U))(127, 0),
            io.axi.r.bits.data
          )
          state  := respond
        }
      }
    }
  }
  when(io.resp.fire) {
    when(state === failed || progress +& 1.U === total)(state := idle).otherwise {
      val next = Mux(
        command.is_2d,
        Mux(
          column +& 1.U === command.tile_width,
          address + (command.source_width - command.tile_width +& 1.U) * command.pixel_bytes,
          address + command.pixel_bytes
        ),
        Mux(
          group +& 1.U === command.groups,
          address + 16.U + command.groups * (command.stride - 1.U) * 16.U,
          address + 16.U
        )
      )
      when(command.is_2d)(column := Mux(column +& 1.U === command.tile_width, 0.U, column + 1.U))
        .otherwise(group := Mux(group +& 1.U === command.groups, 0.U, group + 1.U))
      address := next; progress := progress + 1.U; second := false.B
      state   := Mux(remaining =/= 0.U, receive, plan)
    }
  }
}
