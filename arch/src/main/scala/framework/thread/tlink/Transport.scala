package framework.thread.tlink

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.acc.memdomain.tomem.configs.{Params => ModuleParam}
import framework.acc.memdomain.toddr._
import framework.core.memory.access.preflight.configs.PmpParams
import framework.ip.bus.axi4

/** Thread transport plans virtual bursts; its core selects storage and translates DDR accesses. */
@instantiable
class Transport(p: ModuleParam, permission: PmpParams) extends Module {

  @public val io = IO(new Bundle {
    val dma  = Flipped(new DmaPort(p, permission))
    val link = new Dma(p.axi)
  })

  val reader = Instantiate(new ReadTransport(p))
  val writer = Instantiate(new WriteTransport(p, permission))
  val stream = Instantiate(new Stream(p.axi))
  reader.io.req.valid    := io.dma.read.valid
  reader.io.req.bits     := io.dma.read.bits.transfer
  io.dma.read.ready      := reader.io.req.ready
  io.dma.readResult <> reader.io.resp
  writer.io.req <> io.dma.write
  writer.io.data <> io.dma.writeData
  io.dma.writeResult <> writer.io.resp
  io.dma.readBusy        := reader.io.busy
  io.dma.writeBusy       := writer.io.busy
  stream.io.acc.ar <> reader.io.axi.ar
  reader.io.axi.r <> stream.io.acc.r
  stream.io.acc.aw <> writer.io.axi.aw
  stream.io.acc.w <> writer.io.axi.w
  writer.io.axi.b <> stream.io.acc.b
  reader.io.axi.aw.ready := false.B
  reader.io.axi.w.ready  := false.B
  reader.io.axi.b.valid  := false.B
  reader.io.axi.b.bits   := 0.U.asTypeOf(reader.io.axi.b.bits)
  writer.io.axi.ar.ready := false.B
  writer.io.axi.r.valid  := false.B
  writer.io.axi.r.bits   := 0.U.asTypeOf(writer.io.axi.r.bits)
  stream.io.writeOffset  := writer.io.offset
  stream.io.writeBytes   := writer.io.bytes
  io.link <> stream.io.dma
}

@instantiable
class ReadTransport(val b: ModuleParam) extends Module {
  val p = b

  val axiParams = p.axi.copy(addressBits = 64)

  require(axiParams.dataBits == 128)

  @public
  val io = IO(new Bundle {
    val req  = Flipped(Decoupled(new ReadTransfer))
    val resp =
      Decoupled(new ReadResult(b))
    val busy = Output(Bool())
    val axi  = new axi4.Port(axiParams)

  })

  val (idle :: shape :: divide :: products :: offset :: check :: plan ::
    offer :: receive :: respond :: drain :: failed :: Nil) = Enum(12)
  val state                                                = RegInit(idle)
  val command                                              = Reg(new ReadTransfer)
  val total                                                = Reg(UInt(17.W))
  val progress                                             = RegInit(0.U(17.W))
  val group                                                = RegInit(0.U(6.W))
  val column                                               = RegInit(0.U(4.W))
  val address                                              = Reg(UInt(64.W))
  val burstAddress                                         = Reg(UInt(axiParams.addressBits.W))
  val candidate                                            = Reg(UInt(9.W))
  val remaining                                            = Reg(UInt(9.W))
  val second                                               = RegInit(false.B)
  val firstPart                                            = Reg(UInt(128.W))
  val result                                               = Reg(UInt(128.W))
  val fault                                                = RegInit(0.U.asTypeOf(new DmaStatus))

  io.req.ready             := state === idle
  io.busy                  := state =/= idle
  io.resp.valid            := state === respond || state === failed
  io.resp.bits.data        := Mux(state === failed, 0.U, result)
  io.resp.bits.last        := state === failed || progress +& 1.U === total
  io.resp.bits.addrcounter := progress(15, 0)
  io.resp.bits.fault       := Mux(state === failed, fault, 0.U.asTypeOf(new DmaStatus))
  io.axi.ar.valid          := state === offer
  io.axi.ar.bits           := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.ar.bits.addr      := burstAddress
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
    command := io.req.bits
    address := io.req.bits.vaddr; progress := 0.U; group := 0.U; column := 0.U; second := false.B
    fault   := 0.U.asTypeOf(new DmaStatus)
    val bad = io.req.bits.len === 0.U || count > 65536.U ||
      io.req.bits.groups === 0.U || io.req.bits.stride === 0.U ||
      (io.req.bits.is_2d && (io.req.bits.pixel_bytes === 0.U || io.req.bits.core_width === 0.U ||
        io.req.bits.source_width < io.req.bits.core_width))
    when(bad)(fail(DmaError.Shape.U, io.req.bits.vaddr)).otherwise {
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
  val shapeDivisor     = Mux(command.is_2d, command.core_width.pad(6), command.groups)
  when(state === shape) {
    quotient     := (total - 1.U)(15, 0)
    remainder    := 0.U
    divisor      := shapeDivisor
    divideLeft   := 16.U
    rowStride    := Mux(command.is_2d, command.source_width * command.pixel_bytes, (command.groups * command.stride) << 4)
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
    lastOffset := rowOffset +& columnOffset +& 15.U
    state      := check
  }
  val lastByte         = command.vaddr +& lastOffset
  when(state === check) {
    when(lastByte(64))(fail(DmaError.Shape.U, command.vaddr)).otherwise(state := plan)
  }

  val contiguous = Mux(
    command.is_2d,
    Mux(
      command.pixel_bytes =/= 16.U,
      1.U,
      Mux(command.source_width === command.core_width, total - progress, command.core_width - column)
    ),
    Mux(command.stride === 1.U, total - progress, command.groups - group)
  )

  val pageBeats    = (4096.U(13.W) - address(11, 0)) >> 4
  val commandBeats = Mux(contiguous < (total - progress), contiguous, total - progress)
  val limited      = Mux(commandBeats < 256.U, commandBeats, 256.U)
  when(state === plan) {
    candidate    := Mux(address(3, 0) =/= 0.U, 1.U, Mux(limited < pageBeats, limited, pageBeats))
    burstAddress := address & ~15.U(64.W)
    state        := offer
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
        fault.address := burstAddress
        state         := Mux(badLast || io.axi.r.bits.last, failed, drain)
      }.otherwise {
        remaining    := remaining - 1.U
        burstAddress := burstAddress + 16.U
        when(address(3, 0) =/= 0.U && !second) {
          firstPart := (io.axi.r.bits.data >> (address(3, 0) * 8.U))(127, 0)
          second    := true.B; candidate := 1.U; state := offer
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
      when(command.is_2d)(column := Mux(column +& 1.U === command.core_width, 0.U, column + 1.U))
        .otherwise(group := Mux(group +& 1.U === command.groups, 0.U, group + 1.U))
      address := next; progress := progress + 1.U; second := false.B
      state   := Mux(remaining =/= 0.U, receive, plan)
    }
  }
}

@instantiable
class WriteTransport(val b: ModuleParam, permission: PmpParams) extends Module {
  val p = b

  val axiParams = p.axi.copy(addressBits = 64)

  require(axiParams.dataBits == 128)

  @public
  val io = IO(new Bundle {
    val req    = Flipped(Decoupled(new WriteTransfer(b, permission)))
    val data   =
      Flipped(Decoupled(new WriteData(b)))
    val resp   = Decoupled(new WriteResult)
    val busy   = Output(Bool())
    val offset = Output(UInt(4.W))
    val bytes  = Output(UInt(13.W))
    val axi    = new axi4.Port(axiParams)

  })

  val idle :: plan :: stream :: waitB :: drain :: respond :: Nil = Enum(6)
  val state                                                      = RegInit(idle)
  val command                                                    = Reg(new WriteTransfer(b, permission))
  val endVA                                                      = Reg(UInt(64.W))
  val burstOffset                                                = Reg(UInt(4.W))
  val burstBytes                                                 = Reg(UInt(13.W))
  val cursorVA                                                   = Reg(UInt(64.W))
  val inputTotal                                                 = Reg(UInt(29.W))
  val consumed                                                   = RegInit(0.U(29.W))
  val sourceEnded                                                = RegInit(false.B)
  val remaining                                                  = Reg(UInt(29.W))
  val firstBeat                                                  = RegInit(true.B)
  val beats                                                      = Reg(UInt(9.W))
  val beatIndex                                                  = RegInit(0.U(9.W))
  val awSent                                                     = RegInit(false.B)
  val wDone                                                      = RegInit(false.B)
  val burstAddress                                               = Reg(UInt(axiParams.addressBits.W))
  val carry                                                      = RegInit(0.U(128.W))
  val outputValid                                                = RegInit(false.B)
  val outputData                                                 = Reg(UInt(128.W))
  val outputMask                                                 = Reg(UInt(16.W))
  val outputLast                                                 = Reg(Bool())
  val fault                                                      = RegInit(0.U.asTypeOf(new DmaStatus))
  val offset                                                     = command.vaddr(3, 0)
  val expectedLast                                               = consumed +& 1.U === inputTotal
  val outputRoom                                                 = !outputValid || (io.axi.w.ready && !outputLast)
  val outputIndex                                                = beatIndex + io.axi.w.fire.asUInt
  val outputHead                                                 = firstBeat && !io.axi.w.fire

  io.offset            := burstOffset
  io.bytes             := burstBytes
  io.req.ready         := state === idle
  io.busy              := state =/= idle
  io.data.ready        := (state === stream && !wDone && outputRoom && fault.error === DmaError.None.U && consumed < inputTotal) ||
    (state === drain && !sourceEnded && consumed < inputTotal)
  io.resp.valid        := state === respond
  io.resp.bits.done    := fault.error === DmaError.None.U
  io.resp.bits.fault   := fault
  io.axi.aw.valid      := state === stream && !awSent
  io.axi.aw.bits       := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.aw.bits.id    := 1.U
  io.axi.aw.bits.addr  := burstAddress
  io.axi.aw.bits.len   := beats - 1.U
  io.axi.aw.bits.size  := 4.U
  io.axi.aw.bits.burst := 1.U
  io.axi.w.valid       := state === stream && outputValid
  io.axi.w.bits.data   := outputData
  io.axi.w.bits.strb   := outputMask
  io.axi.w.bits.last   := outputLast
  io.axi.b.ready       := state === waitB
  io.axi.ar.valid      := false.B
  io.axi.ar.bits       := 0.U.asTypeOf(new axi4.Address(axiParams))
  io.axi.r.ready       := false.B

  def terminate(error: UInt, at: UInt): Unit = {
    fault.error   := error
    fault.address := at
    state         := Mux(sourceEnded || consumed === inputTotal, respond, drain)
  }

  when(io.req.fire) {
    command  := io.req.bits
    consumed := 0.U; sourceEnded := false.B; firstBeat := true.B
    carry    := 0.U; outputValid := false.B; fault     := 0.U.asTypeOf(new DmaStatus)
    val finalByte = io.req.bits.vaddr +& (io.req.bits.len - 1.U)
    val malformed = io.req.bits.len === 0.U || io.req.bits.len(3, 0) =/= 0.U || finalByte(64)
    endVA             := finalByte(63, 0)
    inputTotal        := io.req.bits.len >> 4
    remaining         := (io.req.bits.len >> 4) +& (io.req.bits.vaddr(3, 0) =/= 0.U).asUInt
    cursorVA          := io.req.bits.vaddr & ~15.U(64.W)
    when(malformed) {
      fault.error := DmaError.Shape.U; fault.address := io.req.bits.vaddr; state := respond
    }.otherwise(state := plan)
  }

  when(state === plan) {
    val pageBeats = (4096.U(13.W) - cursorVA(11, 0)) >> 4
    val capped    = Mux(remaining < 256.U, remaining, 256.U)
    val extent    = Mux(capped < pageBeats, capped, pageBeats)
    beats := extent; beatIndex := 0.U; awSent := false.B; wDone := false.B
    val head         = Mux(firstBeat, offset, 0.U)
    val covered      = (extent << 4) - head
    val commandBytes = endVA - (cursorVA + head) + 1.U
    burstOffset  := head
    burstBytes   := Mux(commandBytes < covered, commandBytes, covered)
    burstAddress := cursorVA
    state        := stream
  }
  when(io.axi.aw.fire)(awSent := true.B)

  // An accepted old W can be replaced immediately; refills take priority over clearing valid.
  when(io.axi.w.fire)(outputValid                                         := false.B)
  when(io.data.fire) {
    val properLast = io.data.bits.last === expectedLast
    assert(properLast, "WriteDma input LAST does not match command length")
    consumed                                            := consumed + 1.U
    when(io.data.bits.last || expectedLast)(sourceEnded := true.B)
    when(state === stream) {
      outputData  := Mux(
        properLast,
        Mux(offset === 0.U, io.data.bits.data, carry | (io.data.bits.data << (offset * 8.U))(127, 0)),
        0.U
      )
      outputMask  := Mux(properLast, Mux(outputHead, ("hffff".U(16.W) << offset)(15, 0), "hffff".U), 0.U)
      outputLast  := outputIndex +& 1.U === beats
      outputValid := true.B
      carry       := (io.data.bits.data >> ((16.U - offset) * 8.U))(127, 0)
      when(!properLast && fault.error === DmaError.None.U) {
        fault.error   := DmaError.Protocol.U
        fault.address := command.vaddr + (consumed << 4)
      }
    }
  }
  when(state === stream && !wDone && outputRoom) {
    when(fault.error =/= DmaError.None.U) {
      // Complete an already accepted AW even if the producer ended early.
      outputData := 0.U; outputMask                           := 0.U
      outputLast := outputIndex +& 1.U === beats; outputValid := true.B
    }.elsewhen(consumed === inputTotal) {
      outputData := carry
      outputMask := ((1.U(17.W) << offset) - 1.U)(15, 0)
      outputLast := outputIndex +& 1.U === beats; outputValid := true.B
    }
  }
  when(io.axi.w.fire) {
    firstBeat              := false.B
    cursorVA               := cursorVA + 16.U; remaining := remaining - 1.U
    beatIndex              := beatIndex + 1.U
    when(outputLast)(wDone := true.B)
  }
  when(state === stream && (awSent || io.axi.aw.fire) &&
    (wDone || (io.axi.w.fire && outputLast)))(state                       := waitB)
  when(io.axi.b.fire) {
    val bad = io.axi.b.bits.id =/= 1.U || io.axi.b.bits.resp =/= 0.U
    when(bad && fault.error === DmaError.None.U) {
      fault.error   := Mux(
        io.axi.b.bits.id =/= 1.U || io.axi.b.bits.resp === 1.U,
        DmaError.Protocol.U,
        DmaError.AccessFault.U
      )
      fault.address := burstAddress
    }
    when(bad || fault.error =/= DmaError.None.U) {
      state := Mux(sourceEnded || consumed === inputTotal, respond, drain)
    }.elsewhen(remaining === 0.U)(state := respond).otherwise(state := plan)
  }
  when(state === drain && (sourceEnded || consumed === inputTotal))(state := respond)
  when(io.resp.fire)(state                                                := idle)
}
