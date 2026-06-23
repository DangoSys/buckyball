package framework.memdomain.frontend.mem.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XRD}

import framework.frontend.BootAddress
import framework.memdomain.frontend.mem.tlb.BBTLBIO
import framework.top.GlobalConfig

class BBReadRequest extends Bundle {
  val vaddr  = UInt(64.W)
  val len    = UInt(32.W)
  val status = new MStatus
  val stride = UInt(19.W)
  val groups = UInt(6.W)
}

class BBReadResponse(dataWidth: Int) extends Bundle {
  val data        = UInt(dataWidth.W)
  val last        = Bool()
  val addrcounter = UInt(16.W)
}

@instantiable
class StreamReader(val b: GlobalConfig)(edge: TLEdgeOut) extends Module {
  // one packet can delivery how many bits
  val beatBits      = b.memDomain.dma_buswidth
  // one packet can delivery how many bytes
  val beatBytes     = beatBits / 8
  // beatBytes x beat times
  val burstMaxBytes = b.memDomain.dma_burst_maxbytes
  val lgBeat        = log2Ceil(beatBytes)

  require(isPow2(beatBytes), s"dma_buswidth bytes must be a power of two, got $beatBytes")
  require(isPow2(burstMaxBytes), s"dma_burst_maxbytes must be a power of two, got $burstMaxBytes")
  require(burstMaxBytes >= beatBytes, s"dma_burst_maxbytes ($burstMaxBytes) must be >= beatBytes ($beatBytes)")

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
  val zeroBuffer: Instance[BootZeroBuffer] = Instantiate(new BootZeroBuffer(b, beatBits, beatBytes))
  val zeroActive = RegInit(false.B)

  val bytesRequested = RegInit(0.U(32.W))
  val bytesReceived  = RegInit(0.U(32.W))

  val inflight = RegInit(false.B)

  val beatIdx    = bytesRequested >> log2Ceil(beatBytes)
  val rowIdx     = beatIdx / reqReg.groups
  val groupIdx   = beatIdx % reqReg.groups
  val readOffset = (rowIdx * reqReg.groups * reqReg.stride + groupIdx) * beatBytes.U
  val read_vaddr = reqReg.vaddr + readOffset

  val bytesLeft       = reqReg.len - bytesRequested
  val groupsLeftInRow = reqReg.groups - groupIdx
  val rowBytesLeft    = groupsLeftInRow * beatBytes.U
  val pageBytesLeft   = (1.U << b.core.pgIdxBits) - read_vaddr(b.core.pgIdxBits - 1, 0)
  val maxBurstBytes   = Seq(bytesLeft, rowBytesLeft, pageBytesLeft, burstMaxBytes.U).reduce((a, c) => Mux(a < c, a, c))

  val burstCandidates = Iterator.iterate(beatBytes)(_ * 2).takeWhile(_ <= burstMaxBytes).toSeq

  val (readBytes, readLgSize) = burstCandidates.foldLeft((beatBytes.U(32.W), lgBeat.U)) {
    case ((bestBytes, bestLg), size) =>
      val lgSize  = log2Ceil(size)
      val aligned = if (size == 1) true.B else read_vaddr(lgSize - 1, 0) === 0.U
      val fits    = maxBurstBytes >= size.U
      (Mux(fits && aligned, size.U, bestBytes), Mux(fits && aligned, lgSize.U, bestLg))
  }

  val get = edge.Get(
    fromSource = 0.U,
    toAddress = 0.U,
    lgSize = readLgSize
  )._2

  io.tlb.req.valid :=
    (state === s_run) &&
      !zeroActive &&
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
      !inflight && !zeroActive && state =/= s_idle

  io.tl.a.bits         := get
  io.tl.a.bits.address := io.tlb.resp.bits.paddr

  io.tlb.resp.ready := io.tl.a.ready && !inflight && !zeroActive

  when(io.tl.a.fire) {
    inflight       := true.B
    bytesRequested := bytesRequested + readBytes
  }

  //------------------------------------------------------------
  // TL D → Response
  //------------------------------------------------------------

  io.tl.d.ready := io.resp.ready && !zeroActive

  io.resp.valid     := Mux(zeroActive, zeroBuffer.io.resp.valid, io.tl.d.valid)
  io.resp.bits.data := Mux(zeroActive, zeroBuffer.io.resp.bits.data, io.tl.d.bits.data)

  val beatCountResp = bytesReceived >> log2Ceil(beatBytes)
  io.resp.bits.addrcounter := Mux(zeroActive, zeroBuffer.io.resp.bits.addrcounter, beatCountResp)

  io.resp.bits.last := Mux(zeroActive, zeroBuffer.io.resp.bits.last, bytesReceived + beatBytes.U >= reqReg.len)

  zeroBuffer.io.resp.ready := io.resp.ready && zeroActive

  when(io.tl.d.fire) {
    inflight      := !edge.last(io.tl.d)
    bytesReceived := bytesReceived + beatBytes.U
  }

  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

  val reqIsZero = BootAddress.isZeroBase(io.req.bits.vaddr)
  zeroBuffer.io.req.valid := io.req.valid && state === s_idle && reqIsZero
  zeroBuffer.io.req.bits  := io.req.bits

  io.req.ready := (state === s_idle) && Mux(reqIsZero, zeroBuffer.io.req.ready, true.B)

  io.busy := (state =/= s_idle) || inflight || zeroBuffer.io.busy

  when(io.req.fire && reqIsZero) {
    zeroActive := true.B
    state      := s_run
  }.elsewhen(io.req.fire) {
    reqReg         := io.req.bits
    bytesRequested := 0.U
    bytesReceived  := 0.U
    inflight       := false.B
    state          := s_run
  }

  when(zeroActive && zeroBuffer.io.resp.fire && zeroBuffer.io.resp.bits.last) {
    zeroActive := false.B
    state      := s_idle
  }

  when(state === s_run && !zeroActive && bytesReceived >= reqReg.len) {
    state := s_idle
  }
}
