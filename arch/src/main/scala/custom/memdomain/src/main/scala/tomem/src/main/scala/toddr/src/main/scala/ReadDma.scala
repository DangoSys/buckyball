package framework.custom.memdomain.toddr

import framework.generic.acc.memdomain.tomem.configs.{Params => ModuleParam}
import framework.generic.contract.memdomain.toddr.DmaError
import framework.custom.memdomain.toddr.ReadResult
import framework.custom.memdomain.toddr.ReadTransfer
import framework.custom.memdomain.toddr.WriteData
import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.hart.memory.access.preflight.{Error, MapQuery, MapResult}
import framework.generic.bus.axi4
import framework.generic.contract.memdomain.toddr.DmaStatus

@instantiable
class ReadDma(val b: ModuleParam) extends Module {
  val p = b

  val prepared  = p.mapping
  val axiParams = p.axi

  require(axiParams.dataBits == 128)
  require(prepared.beatBytes == 16 && prepared.bus.addressBits == axiParams.addressBits)

  @public
  val io = IO(new Bundle {
    val req           = Flipped(Decoupled(new ReadTransfer))
    val resp          =
      Decoupled(new ReadResult(b))
    val parentId      = Input(UInt(prepared.idBits.W))
    val decisionValid = Input(Bool())
    val decisionFault = Input(new DmaStatus)
    val busy          = Output(Bool())

    val query = Output(new MapQuery(b.mapping))

    val mapping = Input(new MapResult(b.mapping))

    val virtualAddress = Output(UInt(64.W))
    val axi            = new axi4.Port(axiParams)

  })

  val (idle :: shape :: divide :: products :: offset :: check :: plan :: query :: lookup :: answer :: validate ::
    offer :: receive :: drain :: failed :: Nil) = Enum(15)
  val state                                     = RegInit(idle)
  val command                                   = Reg(new ReadTransfer)
  val blockRows                                 = (1.U(6.W) << command.block_log2)(5, 0)
  val blockMask                                 = blockRows - 1.U
  val parent                                    = Reg(UInt(prepared.idBits.W))
  val total                                     = Reg(UInt(17.W))
  val progress                                  = RegInit(0.U(17.W))
  val group                                     = RegInit(0.U(6.W))
  val column                                    = RegInit(0.U(4.W))
  val address                                   = Reg(UInt(64.W))
  val queryVA                                   = Reg(UInt(64.W))
  val physical                                  = Reg(UInt(axiParams.addressBits.W))
  val candidate                                 = Reg(UInt(9.W))
  val remaining                                 = Reg(UInt(9.W))
  val second                                    = RegInit(false.B)
  val firstPart                                 = Reg(UInt(128.W))
  val streaming                                 = RegInit(false.B)
  val result                                    = Reg(UInt(128.W))
  val resultValid                               = RegInit(false.B)
  val resultCounter                             = Reg(UInt(16.W))
  val resultLast                                = Reg(Bool())
  val fault                                     = RegInit(0.U.asTypeOf(new DmaStatus))
  val mapped                                    = Reg(new MapResult(b.mapping))

  io.virtualAddress        := queryVA
  io.req.ready             := state === idle && !resultValid && io.decisionValid
  io.busy                  := state =/= idle || resultValid
  io.resp.valid            := resultValid || state === failed
  io.resp.bits.data        := Mux(resultValid, result, 0.U)
  io.resp.bits.last        := Mux(resultValid, resultLast, true.B)
  io.resp.bits.addrcounter := resultCounter
  io.resp.bits.fault       := Mux(resultValid, 0.U.asTypeOf(new DmaStatus), fault)
  io.query                 := 0.U.asTypeOf(new MapQuery(b.mapping))
  // The prepared map answers a query held for three cycles.
  io.query.valid           := state === query || state === lookup || state === answer
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
  io.axi.r.ready           := (state === receive && (!resultValid || io.resp.ready)) || state === drain
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
      (io.req.bits.is_strided && (io.req.bits.byte_stride === 0.U ||
        io.req.bits.row_bytes === 0.U || io.req.bits.row_bytes > 16.U ||
        io.req.bits.block_log2 > 4.U ||
        (count & ((1.U(17.W) << io.req.bits.block_log2) - 1.U)) =/= 0.U ||
        (io.req.bits.block_log2 =/= 0.U && (io.req.bits.row_bytes =/= 16.U ||
          io.req.bits.byte_stride < (16.U(13.W) << io.req.bits.block_log2))))) ||
      (io.req.bits.is_2d && (io.req.bits.pixel_bytes === 0.U || io.req.bits.core_width === 0.U ||
        io.req.bits.source_width < io.req.bits.core_width))
    when(io.decisionFault.error =/= DmaError.None.U) {
      fail(io.decisionFault.error, io.decisionFault.address)
    }.elsewhen(bad)(fail(DmaError.Shape.U, io.req.bits.vaddr)).otherwise {
      total := count(16, 0); state := shape
    }
  }

  // At most 65536 beats: divide the 16-bit last index without a combinational divider.
  val quotient         = Reg(UInt(16.W))
  val remainder        = Reg(UInt(6.W))
  val divisor          = Reg(UInt(6.W))
  val divideLeft       = Reg(UInt(5.W))
  val rowStride        = Reg(UInt(29.W))
  val columnStride     = Reg(UInt(10.W))
  val rowOffset        = Reg(UInt(45.W))
  val columnOffset     = Reg(UInt(16.W))
  val lastOffset       = Reg(UInt(46.W))
  val shapeDivisor     = Mux(command.is_strided, blockRows, Mux(command.is_2d, command.core_width.pad(6), command.groups))
  when(state === shape) {
    quotient     := (total - 1.U)(15, 0)
    remainder    := 0.U
    divisor      := shapeDivisor
    divideLeft   := 16.U
    rowStride    := Mux(
      command.is_strided,
      command.byte_stride,
      Mux(command.is_2d, command.source_width * command.pixel_bytes, (command.groups * command.stride) << 4)
    )
    columnStride := Mux(command.is_2d, command.pixel_bytes, 16.U)
    state        := Mux(shapeDivisor === 1.U, products, divide)
  }
  val shiftedRemainder = Cat(remainder, quotient(15))
  val subtractDivisor  = shiftedRemainder >= divisor
  when(state === divide) {
    quotient                       := Cat(quotient(14, 0), subtractDivisor)
    remainder                      := Mux(subtractDivisor, shiftedRemainder - divisor, shiftedRemainder)(5, 0)
    divideLeft                     := divideLeft - 1.U
    when(divideLeft === 1.U)(state := products)
  }
  when(state === products) {
    rowOffset    := quotient * rowStride
    columnOffset := remainder * columnStride
    state        := offset
  }
  when(state === offset) {
    lastOffset := rowOffset +& columnOffset +& Mux(command.is_strided, command.row_bytes - 1.U, 15.U)
    state      := check
  }
  val lastByte         = command.vaddr +& lastOffset
  when(state === check) {
    when(lastByte(64))(fail(DmaError.Shape.U, command.vaddr)).otherwise(state := plan)
  }

  val contiguous = Mux(
    command.is_strided,
    Mux(
      command.row_bytes === 16.U,
      Mux(command.byte_stride === (blockRows << 4), total - progress, blockRows - (progress & blockMask)),
      1.U
    ),
    Mux(
      command.is_2d,
      Mux(
        command.pixel_bytes =/= 16.U,
        1.U,
        Mux(command.source_width === command.core_width, total - progress, command.core_width - column)
      ),
      Mux(command.stride === 1.U, total - progress, command.groups - group)
    )
  )

  val pageBeats     = (4096.U(13.W) - Cat(address(11, 4), 0.U(4.W))) >> 4
  val commandBeats  = Mux(contiguous < (total - progress), contiguous, total - progress)
  val shiftedStream = address(3, 0) =/= 0.U && commandBeats > 1.U && pageBeats > 1.U &&
    (!command.is_strided || command.row_bytes === 16.U)
  val requested     = commandBeats +& shiftedStream.asUInt
  val limited       = Mux(requested < 256.U, requested, 256.U)
  when(state === plan) {
    queryVA   := address & ~15.U(64.W)
    streaming := shiftedStream
    candidate := Mux(address(3, 0) =/= 0.U && !shiftedStream, 1.U, Mux(limited < pageBeats, limited, pageBeats))
    state     := query
  }
  when(state === query)(state := lookup)
  when(state === lookup)(state := answer)
  when(state === answer) { mapped := io.mapping; state := validate }
  when(state === validate) {
    when(!mapped.hit || mapped.error =/= Error.Ok.U) {
      val error = MuxLookup(mapped.error, DmaError.AccessFault.U)(Seq(
        Error.Shape.U     -> DmaError.Shape.U,
        Error.Overflow.U  -> DmaError.Shape.U,
        Error.PageFault.U -> DmaError.PageFault.U,
        Error.Capacity.U  -> DmaError.Capacity.U,
        Error.Context.U   -> DmaError.Context.U
      ))
      fail(error, queryVA)
    }.elsewhen(mapped.pa(3, 0) =/= 0.U || (mapped.pa(11, 0) +& (candidate << 4)) > 4096.U) {
      fail(DmaError.AccessFault.U, queryVA)
    }.otherwise { physical := mapped.pa; state := offer }
  }
  when(io.resp.fire) {
    resultValid              := false.B
    when(!resultValid)(state := idle)
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
        when(address(3, 0) =/= 0.U && !second &&
          (!command.is_strided || address(3, 0) +& command.row_bytes > 16.U)) {
          firstPart := (io.axi.r.bits.data >> (address(3, 0) * 8.U))(127, 0)
          second    := true.B
          when(!streaming) {
            queryVA := queryVA + 16.U; candidate := 1.U; state := query
          }
        }.otherwise {
          val finished = progress +& 1.U === total
          result             := Mux(
            second,
            firstPart | (io.axi.r.bits.data << ((16.U - address(3, 0)) * 8.U))(127, 0),
            (io.axi.r.bits.data >> (address(3, 0) * 8.U))(127, 0)
          )
          resultValid        := true.B
          resultCounter      := progress(15, 0)
          resultLast         := finished
          address            := Mux(
            command.is_strided,
            Mux((progress & blockMask) === blockMask, address + command.byte_stride - (blockMask << 4), address + 16.U),
            Mux(
              command.is_2d,
              Mux(
                column +& 1.U === command.core_width,
                address + (command.source_width - command.core_width +& 1.U) * command.pixel_bytes,
                address + command.pixel_bytes
              ),
              Mux(
                group +& 1.U === command.groups,
                address + 16.U + command.groups * (command.stride - 1.U) * 16.U,
                address + 16.U
              )
            )
          )
          when(command.is_2d)(column := Mux(column +& 1.U === command.core_width, 0.U, column + 1.U))
            .otherwise(group := Mux(group +& 1.U === command.groups, 0.U, group + 1.U))
          progress           := progress + 1.U
          // Carry the overlapping bytes within this burst; no data survives a burst.
          firstPart          := (io.axi.r.bits.data >> (address(3, 0) * 8.U))(127, 0)
          second             := streaming && remaining =/= 1.U
          state              := Mux(finished, idle, Mux(remaining === 1.U, plan, receive))
        }
      }
    }
  }
}
