package framework.memdomain.frontend.outside_channel.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XRD}

import framework.memdomain.frontend.outside_channel.tlb.BBTLBIO
import framework.top.GlobalConfig

class BBReadRequest extends Bundle {
  val vaddr  = UInt(64.W)
  val len    = UInt(16.W)
  val status = new MStatus
  val stride = UInt(10.W) // 暂时不用
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(10.W)
}

@instantiable
class StreamReader(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {

  val beatBits  = b.memDomain.dma_buswidth
  val beatBytes = beatBits / 8

  @public
  val io = IO(new Bundle {
    val req   = Flipped(Decoupled(new BBReadRequest()))
    val resp  = Decoupled(new BBReadResponse(beatBits))
    val tlb   = Flipped(new BBTLBIO(b))
    val busy  = Output(Bool())
    val flush = Input(Bool())
    val tl    = new TLBundle(edge.bundle)
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

  val get = edge.Get(
    fromSource = 0.U,
    toAddress = 0.U,
    lgSize = log2Ceil(beatBytes).U
  )._2

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

  io.tl.a.valid :=
    io.tlb.resp.valid && !io.tlb.resp.bits.miss &&
      !inflight && state =/= s_idle

  io.tl.a.bits         := get
  io.tl.a.bits.address := io.tlb.resp.bits.paddr

  io.tlb.resp.ready := io.tl.a.ready && !inflight

  when(io.tl.a.fire) {
    inflight       := true.B
    bytesRequested := bytesRequested + beatBytes.U
  }

  //------------------------------------------------------------
  // TL D → Response
  //------------------------------------------------------------

  io.tl.d.ready := io.resp.ready

  io.resp.valid     := io.tl.d.valid
  io.resp.bits.data := io.tl.d.bits.data

  val beatCountResp = bytesReceived >> log2Ceil(beatBytes)
  io.resp.bits.addrcounter := beatCountResp(9, 0)

  io.resp.bits.last :=
    (bytesReceived + beatBytes.U >= reqReg.len)

  when(io.tl.d.fire) {
    inflight      := false.B
    bytesReceived := bytesReceived + beatBytes.U
  }

  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

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
