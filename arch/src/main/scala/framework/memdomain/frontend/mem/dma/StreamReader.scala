package framework.memdomain.frontend.mem.dma

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import freechips.rocketchip.tilelink._
import freechips.rocketchip.rocket.{MStatus, M_XRD}

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

  val bytesRequested = RegInit(0.U(32.W))
  val bytesReceived  = RegInit(0.U(32.W))
  val inflight       = RegInit(false.B)
  val unalignedTxn   = RegInit(false.B)
  val readSecond     = RegInit(false.B)
  val firstData      = RegInit(0.U(beatBits.W))
  val respValid      = RegInit(false.B)
  val respData       = RegInit(0.U(beatBits.W))
  val groupIdx       = RegInit(0.U(6.W))
  val readVaddr      = RegInit(0.U(64.W))
  val rowSkip        = RegInit(0.U(32.W))

  val addrOffset = if (beatBytes == 1) 0.U(1.W) else readVaddr(lgBeat - 1, 0)

  val alignedReadVaddr =
    if (beatBytes == 1) {
      readVaddr
    } else {
      Cat(readVaddr(63, lgBeat), 0.U(lgBeat.W))
    }

  val secondReadVaddr = alignedReadVaddr + beatBytes.U
  val needUnaligned   = if (beatBytes == 1) false.B else addrOffset =/= 0.U
  val issueVaddr      = Mux(needUnaligned, Mux(readSecond, secondReadVaddr, alignedReadVaddr), readVaddr)

  val bytesLeft       = reqReg.len - bytesRequested
  val groupsLeftInRow = reqReg.groups - groupIdx
  val rowBytesLeft    = groupsLeftInRow << lgBeat
  val pageBytesLeft   = (1.U << b.core.pgIdxBits) - readVaddr(b.core.pgIdxBits - 1, 0)
  val maxBurstBytes   = Seq(bytesLeft, rowBytesLeft, pageBytesLeft, burstMaxBytes.U).reduce((a, c) => Mux(a < c, a, c))

  val burstCandidates = Iterator.iterate(beatBytes)(_ * 2).takeWhile(_ <= burstMaxBytes).toSeq

  val (readBytes, readLgSize) = burstCandidates.foldLeft((beatBytes.U(32.W), lgBeat.U)) {
    case ((bestBytes, bestLg), size) =>
      val lgSize  = log2Ceil(size)
      val aligned = if (size == 1) true.B else readVaddr(lgSize - 1, 0) === 0.U
      val fits    = maxBurstBytes >= size.U
      (Mux(fits && aligned, size.U, bestBytes), Mux(fits && aligned, lgSize.U, bestLg))
  }

  val get = edge.Get(
    fromSource = 0.U,
    toAddress = 0.U,
    lgSize = Mux(needUnaligned, lgBeat.U, readLgSize)
  )._2

  io.tlb.req.valid :=
    (state === s_run) &&
      (bytesRequested < reqReg.len) &&
      !inflight &&
      !respValid

  io.tlb.req.bits             := DontCare
  io.tlb.req.bits.vaddr       := issueVaddr
  io.tlb.req.bits.passthrough := false.B
  io.tlb.req.bits.size        := 0.U
  io.tlb.req.bits.cmd         := M_XRD
  io.tlb.req.bits.prv         := 3.U
  io.tlb.req.bits.v           := false.B
  io.tlb.req.bits.status      := reqReg.status

  val tlbFault = io.tlb.resp.valid && !io.tlb.resp.bits.miss &&
    (io.tlb.resp.bits.pf.ld || io.tlb.resp.bits.ae.ld || io.tlb.resp.bits.gf.ld)
  val tlbOk    = io.tlb.resp.valid && !io.tlb.resp.bits.miss && !tlbFault

  assert(!(state === s_run && tlbFault && !inflight && !respValid), "DMA load TLB fault")

  io.tl.a.valid :=
    tlbOk && !inflight && !respValid && state =/= s_idle

  io.tl.a.bits         := get
  io.tl.a.bits.address := io.tlb.resp.bits.paddr

  io.tlb.resp.ready := io.tl.a.ready && !inflight && !respValid

  when(io.tl.a.fire) {
    inflight     := true.B
    unalignedTxn := needUnaligned
    when(!needUnaligned) {
      bytesRequested := bytesRequested + readBytes
      val nextG = groupIdx + (readBytes >> lgBeat)
      assert(nextG <= reqReg.groups, "StreamReader burst crossed row")
      when(nextG === reqReg.groups) {
        groupIdx  := 0.U
        readVaddr := readVaddr + readBytes + (rowSkip << lgBeat)
      }.otherwise {
        groupIdx  := nextG
        readVaddr := readVaddr + readBytes
      }
    }
  }

  //------------------------------------------------------------
  // TL D → Response
  //------------------------------------------------------------

  io.tl.d.ready := inflight && Mux(unalignedTxn, !respValid, io.resp.ready)

  val firstBytes = beatBytes.U - addrOffset
  val mergedData = (io.tl.d.bits.data << (firstBytes * 8.U)) |
    (firstData >> (addrOffset * 8.U))

  io.resp.valid     := Mux(respValid, true.B, io.tl.d.valid && !unalignedTxn)
  io.resp.bits.data := Mux(respValid, respData, io.tl.d.bits.data)

  val beatCountResp = bytesReceived >> lgBeat
  io.resp.bits.addrcounter := beatCountResp

  val lastResp = bytesReceived + beatBytes.U >= reqReg.len
  io.resp.bits.last := lastResp

  when(io.tl.d.fire) {
    when(unalignedTxn) {
      inflight := false.B
      when(!readSecond) {
        firstData  := io.tl.d.bits.data
        readSecond := true.B
      }.otherwise {
        respData     := mergedData
        respValid    := true.B
        readSecond   := false.B
        unalignedTxn := false.B
      }
    }.otherwise {
      inflight      := !edge.last(io.tl.d)
      bytesReceived := bytesReceived + beatBytes.U
    }
  }

  when(respValid && io.resp.fire) {
    respValid      := false.B
    bytesRequested := bytesRequested + beatBytes.U
    bytesReceived  := bytesReceived + beatBytes.U
    when(groupIdx + 1.U === reqReg.groups) {
      groupIdx  := 0.U
      readVaddr := readVaddr + beatBytes.U + (rowSkip << lgBeat)
    }.otherwise {
      groupIdx  := groupIdx + 1.U
      readVaddr := readVaddr + beatBytes.U
    }
    state          := Mux(lastResp, s_idle, s_run)
  }

  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.e.valid := false.B

  io.req.ready := state === s_idle

  io.busy := (state =/= s_idle) || inflight || respValid

  when(io.req.fire) {
    assert(io.req.bits.groups >= 1.U, "StreamReader groups must be >= 1")
    assert(io.req.bits.stride >= 1.U, "StreamReader stride must be >= 1")
    reqReg         := io.req.bits
    bytesRequested := 0.U
    bytesReceived  := 0.U
    inflight       := false.B
    unalignedTxn   := false.B
    readSecond     := false.B
    respValid      := false.B
    firstData      := 0.U
    respData       := 0.U
    groupIdx       := 0.U
    readVaddr      := io.req.bits.vaddr
    rowSkip        := io.req.bits.groups * (io.req.bits.stride - 1.U)
    state          := Mux(io.req.bits.len === 0.U, s_idle, s_run)
  }

  when(state === s_run && bytesReceived >= reqReg.len) {
    state := s_idle
  }
}
