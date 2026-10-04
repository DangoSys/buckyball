package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class ImageLoader(val b: GlobalConfig) extends Module {
  private val p = b.rvv

  @public
  val io = IO(new Bundle {
    val start          = Flipped(Decoupled(new ImageLoad))
    val image          = Flipped(Decoupled(UInt(32.W)))
    val abort          = Input(Bool())
    val program        = Decoupled(new ProgramWrite)
    val memoryRequest  = Decoupled(new VectorMemoryRequest)
    val memoryResponse = Flipped(Decoupled(new VectorMemoryResponse))
    val done           = Decoupled(new ImageResult)
  })

  val idle :: header :: validate :: text :: data :: writeSend :: writeWait :: completed :: Nil = Enum(8)

  val state     = RegInit(idle)
  val fields    = Reg(Vec(6, UInt(32.W)))
  val index     = RegInit(0.U(32.W))
  val buffer    = Reg(Bool())
  val bytes     = Reg(UInt(32.W))
  val offset    = Reg(UInt(32.W))
  val request   = Reg(new VectorMemoryRequest)
  val nextState = Reg(UInt(4.W))
  val aborted   = RegInit(false.B)
  val result    = RegInit(0.U.asTypeOf(new ImageResult))

  io.start.ready          := state === idle
  io.image.ready          := !io.abort && (state === header || state === text && io.program.ready || state === data)
  io.program.valid        := state === text && io.image.valid && !io.abort
  io.program.bits.buffer  := buffer
  io.program.bits.first   := index === 0.U
  io.program.bits.address := index
  io.program.bits.data    := io.image.bits
  io.memoryRequest.valid  := state === writeSend && !io.abort
  io.memoryRequest.bits   := request
  io.memoryResponse.ready := state === writeWait
  io.done.valid           := state === completed
  io.done.bits            := result

  when(io.start.fire) {
    buffer  := io.start.bits.buffer
    bytes   := io.start.bits.bytes
    index   := 0.U
    result  := 0.U.asTypeOf(new ImageResult)
    aborted := false.B
    state   := header
    when(io.start.bits.bytes < 28.U || io.start.bits.bytes(1, 0).orR) {
      result.fault := true.B
      result.cause := 2.U
      result.tval  := io.start.bits.bytes
      state        := completed
    }
  }
  when(state === header && io.image.fire) {
    fields(index(2, 0))       := io.image.bits
    index                     := index + 1.U
    when(index === 5.U)(state := validate)
  }
  when(state === validate && !io.abort) {
    result.entry      := fields(2)
    result.textBytes  := fields(1)
    result.constBytes := fields(4)
    when(fields(0) =/= "h31564b52".U || fields(1) === 0.U || fields(1) > (p.iBufWords * 4).U ||
      fields(1)(1, 0).orR || fields(2)(1, 0).orR || fields(2) >= fields(1) || fields(4)(1, 0).orR ||
      24.U +& fields(1) +& fields(4) =/= bytes) {
      result.fault := true.B
      result.cause := 2.U
      result.tval  := fields(0)
      state        := completed
    }.elsewhen(fields(5) =/= 0.U) {
      result.fault := true.B
      result.cause := 2.U
      result.tval  := fields(5)
      state        := completed
    }.elsewhen(fields(3) =/= "h80000000".U || fields(4) > p.constBytes.U) {
      result.fault := true.B
      result.cause := 2.U
      result.tval  := fields(3)
      state        := completed
    }.otherwise {
      index  := 0.U
      offset := 0.U
      state  := text
    }
  }
  when(state === text && io.program.fire) {
    index := index + 1.U
    when((index + 1.U) * 4.U === fields(1)) {
      offset := 0.U
      state  := Mux(fields(4) =/= 0.U, data, completed)
    }
  }
  when(state === data && io.image.fire) {
    request.address := fields(3) + offset
    request.write   := true.B
    request.data    := io.image.bits
    request.mask    := 15.U
    request.size    := 2.U
    nextState       := Mux(offset + 4.U === fields(4), completed, data)
    state           := writeSend
  }
  when(io.memoryRequest.fire)(state := writeWait)
  when(io.memoryResponse.fire) {
    when(io.memoryResponse.bits.error) {
      result.fault := true.B
      result.cause := 7.U
      result.tval  := request.address
      state        := completed
    }.elsewhen(aborted || io.abort) {
      state := completed
    }.otherwise {
      offset := offset + 4.U
      state  := nextState
    }
  }
  when(io.abort && state =/= idle && state =/= completed) {
    aborted                                                   := true.B
    when(!io.memoryResponse.fire || !io.memoryResponse.bits.error) {
      result.fault := true.B
      result.cause := 5.U
      result.tval  := Mux(state === writeWait || state === writeSend, request.address, 0.U)
    }
    when(state =/= writeWait || io.memoryResponse.fire)(state := completed)
  }
  when(io.done.fire)(state          := idle)
}
