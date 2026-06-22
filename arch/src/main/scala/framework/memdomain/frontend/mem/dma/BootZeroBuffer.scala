package framework.memdomain.frontend.mem.dma

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.top.GlobalConfig

@instantiable
class BootZeroBuffer(val b: GlobalConfig, val dataWidth: Int, val beatBytes: Int) extends Module {

  @public
  val io = IO(new Bundle {
    val req  = Flipped(Decoupled(new BBReadRequest()))
    val resp = Decoupled(new BBReadResponse(dataWidth))
    val busy = Output(Bool())
  })

  private val sIdle :: sRun :: Nil = Enum(2)
  private val state                = RegInit(sIdle)
  private val lenReg               = RegInit(0.U(32.W))
  private val bytesSent            = RegInit(0.U(32.W))

  private val nextBytes = bytesSent + beatBytes.U
  private val lastBeat  = nextBytes >= lenReg
  private val beatCount = bytesSent >> log2Ceil(beatBytes)

  io.req.ready := state === sIdle

  io.resp.valid            := state === sRun
  io.resp.bits.data        := 0.U
  io.resp.bits.addrcounter := beatCount
  io.resp.bits.last        := lastBeat

  io.busy := state =/= sIdle

  when(io.req.fire) {
    lenReg    := io.req.bits.len
    bytesSent := 0.U
    state     := Mux(io.req.bits.len === 0.U, sIdle, sRun)
  }

  when(io.resp.fire) {
    bytesSent := nextBytes
    when(lastBeat) {
      state := sIdle
    }
  }
}
