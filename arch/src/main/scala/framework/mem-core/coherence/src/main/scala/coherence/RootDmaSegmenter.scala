package memcore.memory.coherence

import chisel3._
import chisel3.util._

/** A strided DMA descriptor expressed in bytes. */
class RootDmaDescriptor extends Bundle {
  val addr      = UInt(64.W)
  val bytes     = UInt(32.W)
  val rowBytes  = UInt(32.W)
  val rowStride = UInt(32.W)
}

/** One contiguous segment that never crosses a row or a page. */
class RootDmaSegment extends Bundle {
  val addr  = UInt(64.W)
  val bytes = UInt(32.W)
  val last  = Bool()
}

/**
 * Splits a DMA descriptor before it reaches the ROOT region engine.
 *
 * The producer supplies the total number of useful bytes. Each logical row
 * contains rowBytes useful bytes and starts rowStride bytes after the previous
 * row. No output segment crosses a row, a page, or maxSegmentBytes.
 */
class RootDmaSegmenter(pageBytes: Int = 4096, maxSegmentBytes: Int = 16384) extends Module {
  require(isPow2(pageBytes) && pageBytes >= 8)
  require(isPow2(maxSegmentBytes) && maxSegmentBytes >= 8)

  val io = IO(new Bundle {
    val descriptor = Flipped(Decoupled(new RootDmaDescriptor))
    val segment    = Decoupled(new RootDmaSegment)
  })

  val idle :: run :: Nil = Enum(2)
  val state              = RegInit(idle)
  val currentAddr        = Reg(UInt(64.W))
  val remaining          = Reg(UInt(32.W))
  val rowRemaining       = Reg(UInt(32.W))
  val rowStride          = Reg(UInt(32.W))
  val rowBytes           = Reg(UInt(32.W))

  private def minimum(values: Seq[UInt]): UInt = values.reduce { (left, right) =>
    Mux(left < right, left, right)
  }

  val pageRemaining = pageBytes.U - currentAddr(log2Ceil(pageBytes) - 1, 0)
  val segmentBytes  = minimum(Seq(remaining, rowRemaining, pageRemaining, maxSegmentBytes.U))

  io.descriptor.ready   := state === idle
  io.segment.valid      := state === run
  io.segment.bits.addr  := currentAddr
  io.segment.bits.bytes := segmentBytes
  io.segment.bits.last  := remaining === segmentBytes

  when(io.descriptor.fire) {
    assert(io.descriptor.bits.bytes =/= 0.U, "ROOT DMA descriptor has zero bytes")
    assert(io.descriptor.bits.rowBytes =/= 0.U, "ROOT DMA descriptor has zero rowBytes")
    assert(io.descriptor.bits.rowStride >= io.descriptor.bits.rowBytes, "ROOT DMA rowStride is smaller than rowBytes")
    assert(io.descriptor.bits.bytes % io.descriptor.bits.rowBytes === 0.U, "ROOT DMA bytes must contain whole rows")
    currentAddr                    := io.descriptor.bits.addr
    remaining                      := io.descriptor.bits.bytes
    rowRemaining                   := io.descriptor.bits.rowBytes
    rowStride                      := io.descriptor.bits.rowStride
    rowBytes                       := io.descriptor.bits.rowBytes
    state                          := run
  }

  when(io.segment.fire) {
    val finishesRow = rowRemaining === segmentBytes
    currentAddr  := currentAddr + segmentBytes + Mux(finishesRow, rowStride - rowBytes, 0.U)
    remaining    := remaining - segmentBytes
    rowRemaining := Mux(finishesRow, rowBytes, rowRemaining - segmentBytes)
    when(remaining === segmentBytes) {
      state := idle
    }
  }
}

object EmitRootDmaSegmenter extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(new RootDmaSegmenter(), args)
}
