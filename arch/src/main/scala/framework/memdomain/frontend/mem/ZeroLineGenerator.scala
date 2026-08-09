package framework.memdomain.frontend.mem

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._

class ZeroLineRequest extends Bundle {
  val rows = UInt(32.W)
}

class ZeroLineResponse(dataWidth: Int) extends Bundle {
  val data = UInt(dataWidth.W)
  val row  = UInt(32.W)
  val last = Bool()
}

/**
 * Generates a stream of zero-filled scratchpad rows.
 *
 * This is a local memory write source, not a DMA engine: it has no address
 * translation or TileLink interface. Its caller supplies the destination and
 * consumes each generated row through its normal bank-write path.
 */
@instantiable
class ZeroLineGenerator(val dataWidth: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val req  = Flipped(Decoupled(new ZeroLineRequest))
    val resp = Decoupled(new ZeroLineResponse(dataWidth))
    val busy = Output(Bool())
  })

  private val sIdle :: sRun :: Nil = Enum(2)
  private val state                = RegInit(sIdle)
  private val rowCount             = RegInit(0.U(32.W))
  private val row                  = RegInit(0.U(32.W))

  io.req.ready := state === sIdle

  io.resp.valid     := state === sRun
  io.resp.bits.data := 0.U
  io.resp.bits.row  := row
  io.resp.bits.last := row + 1.U >= rowCount

  io.busy := state =/= sIdle

  when(io.req.fire) {
    rowCount := io.req.bits.rows
    row      := 0.U
    state    := Mux(io.req.bits.rows === 0.U, sIdle, sRun)
  }

  when(io.resp.fire) {
    row := row + 1.U
    when(io.resp.bits.last) {
      state := sIdle
    }
  }
}
