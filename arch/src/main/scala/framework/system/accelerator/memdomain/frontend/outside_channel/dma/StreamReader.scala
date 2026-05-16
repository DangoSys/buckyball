package framework.system.accelerator.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4BundleParameters}
import freechips.rocketchip.rocket.{MStatus, M_XRD}

import framework.system.accelerator.memdomain.frontend.outside_channel.tlb.BBTLBIO
import framework.system.link.BBAxi4Params
import framework.top.GlobalConfig

class BBReadRequest extends Bundle {
  val vaddr  = UInt(64.W)
  val len    = UInt(16.W)
  val status = new MStatus
  val stride = UInt(10.W) // currently unused
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(10.W)
}

@instantiable
class StreamReader(val b: GlobalConfig) extends Module {

  val beatBits  = b.memDomain.dma_buswidth
  val beatBytes = beatBits / 8
  val axiParams: AXI4BundleParameters = BBAxi4Params(b)

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBReadRequest()))
    val resp  = Decoupled(new BBReadResponse(beatBits))
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    val axi   = AXI4Bundle(axiParams)
  })

  //------------------------------------------------------------
  // FSM
  //------------------------------------------------------------

  val s_idle :: s_run :: Nil = Enum(2)
  val state                  = RegInit(s_idle)

  val reqReg = Reg(new BBReadRequest())

  val bytesRequested = RegInit(0.U(16.W))
  val bytesReceived  = RegInit(0.U(16.W))

  val inflight   = RegInit(false.B)
  val read_vaddr = reqReg.vaddr + bytesRequested

  io.tlb.req.valid :=
    (state === s_run) &&
      (bytesRequested < reqReg.len) &&
      !inflight

  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := read_vaddr
  io.tlb.req.bits.passthrough := false.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XRD
  io.tlb.req.bits.prv         := 3.U
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := reqReg.status

  //------------------------------------------------------------
  // AXI4 AR channel (read request)
  //------------------------------------------------------------

  io.axi.ar.valid :=
    io.tlb.resp.valid && !io.tlb.resp.bits.miss &&
      !inflight && state =/= s_idle

  io.axi.ar.bits       := DontCare
  io.axi.ar.bits.id    := 0.U
  io.axi.ar.bits.addr  := io.tlb.resp.bits.paddr
  io.axi.ar.bits.len   := 0.U                   // single beat (len = beats - 1)
  io.axi.ar.bits.size  := log2Ceil(beatBytes).U // bytes/beat = 2^size
  io.axi.ar.bits.burst := 1.U                   // INCR
  io.axi.ar.bits.lock  := 0.U
  io.axi.ar.bits.cache := 0.U
  io.axi.ar.bits.prot  := 0.U
  io.axi.ar.bits.qos   := 0.U

  io.tlb.resp.ready := io.axi.ar.ready && !inflight

  when(io.axi.ar.fire) {
    inflight       := true.B
    bytesRequested := bytesRequested + beatBytes.U
  }

  //------------------------------------------------------------
  // AXI4 R channel -> Response
  //------------------------------------------------------------

  io.axi.r.ready := io.resp.ready

  io.resp.valid     := io.axi.r.valid
  io.resp.bits.data := io.axi.r.bits.data

  val beatCountResp = bytesReceived >> log2Ceil(beatBytes)
  io.resp.bits.addrcounter := beatCountResp(9, 0)

  io.resp.bits.last :=
    (bytesReceived + beatBytes.U >= reqReg.len)

  when(io.axi.r.fire) {
    inflight      := false.B
    bytesReceived := bytesReceived + beatBytes.U
  }

  //------------------------------------------------------------
  // Tie off unused AXI4 channels (write path)
  //------------------------------------------------------------

  io.axi.aw.valid := false.B
  io.axi.aw.bits  := DontCare
  io.axi.w.valid  := false.B
  io.axi.w.bits   := DontCare
  io.axi.b.ready  := true.B

  io.req.ready := (state === s_idle)

  io.busy := (state =/= s_idle) || inflight

  when(io.req.fire) {
    reqReg         := io.req.bits
    bytesRequested := 0.U
    bytesReceived  := 0.U
    inflight       := false.B
    state          := s_run
  }

  when(state === s_run && bytesReceived >= reqReg.len) {
    state := s_idle
  }
}
