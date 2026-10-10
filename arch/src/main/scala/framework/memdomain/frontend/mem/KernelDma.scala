package framework.memdomain.frontend.mem

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.memdomain.frontend.mem.dma.{BBReadRequest, BBReadResponse, DmaError, DmaStatus}
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead, BankWrite}

class KernelDmaRequest(b: GlobalConfig) extends Bundle {
  val rob_id  = UInt(log2Ceil(b.frontend.rob_entries).W)
  val address = UInt(64.W)
  val bytes   = UInt(32.W)
}

class KernelDmaPort(b: GlobalConfig) extends Bundle {
  val load   = Flipped(Decoupled(new KernelDmaRequest(b)))
  val image  = Decoupled(UInt(32.W))
  val result = Decoupled(new DmaStatus)
  val abort  = Input(Bool())
  val busy   = Output(Bool())
}

class KernelMemoryBridge(b: GlobalConfig) extends KernelDmaPort(b) {
  val bankRead  = Vec(b.rvv.memoryPorts, new BankRead(b))
  val bankWrite = Vec(b.rvv.memoryPorts, new BankWrite(b))
  val active    = Input(Bool())
  val ready     = Output(Bool())
}

@instantiable
class KernelDma(val b: GlobalConfig) extends Module {
  val beatWords = b.memDomain.dma_buswidth / 32
  val beatBytes = b.memDomain.dma_buswidth / 8
  require(beatWords > 0 && b.memDomain.dma_buswidth % 32 == 0)

  @public
  val io = IO(new Bundle {
    val footprint = Output(new Footprint(b))
    val kernel    = new KernelDmaPort(b)
    val request   = Decoupled(new BBReadRequest)
    val response  = Flipped(Decoupled(new BBReadResponse(b.memDomain.dma_buswidth)))
  })

  val idle :: request :: receive :: output :: terminal :: Nil = Enum(5)
  val state                                                   = RegInit(idle)
  val descriptor                                              = Reg(new KernelDmaRequest(b))
  val remaining                                               = Reg(UInt(32.W))
  val data                                                    = Reg(UInt(b.memDomain.dma_buswidth.W))
  val words                                                   = Reg(UInt(math.max(1, log2Ceil(beatWords + 1)).W))
  val index                                                   = Reg(UInt(math.max(1, log2Ceil(beatWords)).W))
  val last                                                    = Reg(Bool())
  val discarded                                               = RegInit(false.B)
  val result                                                  = RegInit(0.U.asTypeOf(new DmaStatus))
  val owner                                                   = Reg(UInt(log2Ceil(b.frontend.rob_entries).W))

  def readSpan(bytes: UInt): UInt =
    (bytes.pad(64) + (beatBytes - 1).U) & (~(BigInt(beatBytes) - 1) & ((BigInt(1) << 64) - 1)).U(64.W)

  def invalidShape(address: UInt, bytes: UInt): Bool = {
    val end = address +& (readSpan(bytes) - 1.U)
    bytes === 0.U || bytes(1, 0) =/= 0.U || end(64)
  }

  // The read DMA fills a complete output beat even for a short final payload.
  // The preflight layer then aligns this span around the original VA.
  io.footprint               := 0.U.asTypeOf(new Footprint(b))
  io.footprint.valid         := state =/= idle
  io.footprint.rob_id        := owner
  io.footprint.baseVA        := descriptor.address
  io.footprint.rows          := 1.U
  io.footprint.columns       := 1.U
  io.footprint.spanBytes     := readSpan(descriptor.bytes)
  io.footprint.fault.error   := Mux(invalidShape(descriptor.address, descriptor.bytes), DmaError.Shape.U, DmaError.None.U)
  io.footprint.fault.address := Mux(io.footprint.fault.error =/= 0.U, descriptor.address, 0.U)

  io.kernel.load.ready   := state === idle && !io.kernel.abort
  io.kernel.busy         := state =/= idle
  io.request.valid       := state === request && !io.kernel.abort
  io.request.bits        := 0.U.asTypeOf(io.request.bits)
  io.request.bits.vaddr  := descriptor.address
  io.request.bits.len    := descriptor.bytes
  io.request.bits.groups := 1.U
  io.request.bits.stride := 1.U
  io.response.ready      := state === receive
  io.kernel.image.valid  := state === output && !discarded && !io.kernel.abort
  io.kernel.image.bits   := (data >> (index * 32.U))(31, 0)
  io.kernel.result.valid := state === terminal
  io.kernel.result.bits  := result

  // Abort cancels a load, never an execution. Once offered, the terminal result
  // is immutable even if the image parser subsequently rejects the image.
  when(io.kernel.abort && state =/= idle && state =/= terminal) {
    discarded                     := true.B
    when(result.error === DmaError.None.U) {
      result.error   := DmaError.Cancelled.U
      result.address := descriptor.address
    }
    when(state === request)(state := terminal)
    when(state === output)(state  := Mux(last, terminal, receive))
  }

  when(io.kernel.load.fire) {
    val invalid = invalidShape(io.kernel.load.bits.address, io.kernel.load.bits.bytes)
    owner          := io.kernel.load.bits.rob_id
    descriptor     := io.kernel.load.bits
    remaining      := io.kernel.load.bits.bytes >> 2
    discarded      := false.B
    result.error   := Mux(invalid, DmaError.Shape.U, DmaError.None.U)
    result.address := Mux(invalid, io.kernel.load.bits.address, 0.U)
    state          := Mux(invalid, terminal, request)
  }
  when(io.request.fire)(state       := receive)
  when(io.response.fire) {
    data      := io.response.bits.data
    words     := Mux(remaining < beatWords.U, remaining, beatWords.U)
    index     := 0.U
    last      := io.response.bits.last
    remaining := remaining - Mux(remaining < beatWords.U, remaining, beatWords.U)
    when(io.response.bits.fault.error =/= DmaError.None.U) {
      when(result.error === DmaError.None.U || result.error === DmaError.Cancelled.U) {
        result := io.response.bits.fault
      }
      discarded                         := true.B
      when(io.response.bits.last)(state := terminal)
    }.elsewhen(discarded || io.kernel.abort) {
      when(io.response.bits.last)(state := terminal)
    }.elsewhen(io.response.bits.last =/= (remaining <= beatWords.U) || remaining === 0.U) {
      result.error                      := DmaError.Protocol.U
      result.address                    := descriptor.address + (descriptor.bytes - (remaining << 2))
      discarded                         := true.B
      when(io.response.bits.last)(state := terminal)
    }.otherwise {
      state := output
    }
  }
  when(state === output && io.kernel.image.fire) {
    when(index +& 1.U === words) {
      state := Mux(last, terminal, receive)
    }.otherwise {
      index := index + 1.U
    }
  }
  when(io.kernel.result.fire)(state := idle)
}
