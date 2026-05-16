package framework.system.link

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import freechips.rocketchip.amba.axi4.{AXI4Bundle, AXI4BundleParameters, AXI4Parameters}
import freechips.rocketchip.tilelink.{TLBundle, TLBundleParameters, TLMessages}

/**
 * AXI4 to TileLink protocol bridge (pure Module, no Diplomacy).
 *
 * Converts AXI4 master traffic into a TileLink TL-UH master interface
 * (no coherence: only Get / PutFullData / AccessAck[Data] are used).
 *
 * Architecture:
 *  - One outstanding read transaction (uses TL source ID 0, with bit[0]=0)
 *  - One outstanding write transaction (uses TL source ID 1, with bit[0]=1)
 *  - Read and write paths share the TL A channel via a simple priority
 *    arbiter (read priority).
 *  - Single beat per transaction (AXI4 burst length must be 0, i.e. AxLEN=0).
 *  - The AXI4 ID issued on AR/AW is captured and replayed on R/B so that
 *    the original master sees its ID round-trip correctly.
 *
 * Limitations (deliberate, see file header in commit):
 *  - len > 0 (multi-beat AXI bursts) are NOT supported. AXI4 masters such
 *    as StreamReader/StreamWriter currently emit AxLEN=0, which is fine.
 *  - No AXI4 atomics, no QoS / cache hint translation.
 *  - Single outstanding per direction; throughput is limited to 1
 *    transaction in flight per AR/AW channel at any time.
 */
@instantiable
class AXI4ToTLBridge(
  val axiParams: AXI4BundleParameters,
  val tlParams:  TLBundleParameters)
    extends Module {

  @public
  val io = IO(new Bundle {
    val axi = Flipped(AXI4Bundle(axiParams))
    val tl  = new TLBundle(tlParams)
  })

  // ---------------------------------------------------------------------------
  // Parameters / sanity checks
  // ---------------------------------------------------------------------------
  // We use bit 0 of source to disambiguate read vs. write.
  require(tlParams.sourceBits >= 1, s"AXI4ToTLBridge requires at least 1 TL source bit, got ${tlParams.sourceBits}")
  // Support AXI4 dataBits >= TL dataBits (downconversion)
  // For now, require exact match. Width conversion will be added in future.
  require(
    axiParams.dataBits >= tlParams.dataBits,
    s"AXI4 dataBits (${axiParams.dataBits}) must be >= TL dataBits (${tlParams.dataBits})"
  )

  private val axiBytes = axiParams.dataBits / 8
  private val tlBytes  = tlParams.dataBits / 8

  private val SRC_READ  = 0.U(tlParams.sourceBits.W)
  private val SRC_WRITE = 1.U(tlParams.sourceBits.W)

  // ---------------------------------------------------------------------------
  // Default tie-offs for unused TL channels (TL-UH only uses A/D)
  // ---------------------------------------------------------------------------
  io.tl.b.ready := true.B
  io.tl.c.valid := false.B
  io.tl.c.bits  := DontCare
  io.tl.e.valid := false.B
  io.tl.e.bits  := DontCare

  // ---------------------------------------------------------------------------
  // Read path:  AXI4 AR  ->  TL Get        ;   TL AccessAckData -> AXI4 R
  // ---------------------------------------------------------------------------
  // FSM:  rIdle -> rSendA (waiting to inject Get on A) -> rWaitD (waiting for
  // AccessAckData) -> rSendR (driving R back to AXI master) -> rIdle.
  //
  // We keep the response one cycle in a register so that we can present the
  // AXI R beat irrevocably (AXI requires R.valid stay high until R.ready).
  val rIdle :: rSendA :: rWaitD :: rSendR :: Nil = Enum(4)
  val rState                                     = RegInit(rIdle)

  val rIdReg   = Reg(UInt(axiParams.idBits.W))
  val rAddrReg = Reg(UInt(axiParams.addrBits.W))
  val rSizeReg = Reg(UInt(axiParams.sizeBits.W))

  val rDataReg = Reg(UInt(tlParams.dataBits.W))
  val rRespReg = Reg(UInt(AXI4Parameters.respBits.W))

  // Capture AR
  io.axi.ar.ready := (rState === rIdle)
  when(io.axi.ar.fire) {
    rIdReg   := io.axi.ar.bits.id
    rAddrReg := io.axi.ar.bits.addr
    rSizeReg := io.axi.ar.bits.size
    rState   := rSendA
    // Single-beat enforcement: assert AxLEN=0
    assert(io.axi.ar.bits.len === 0.U, "AXI4ToTLBridge: AR burst length > 0 not supported (got %d)", io.axi.ar.bits.len)
  }

  // ---------------------------------------------------------------------------
  // Write path:  AXI4 AW + W  ->  TL PutFullData   ;   TL AccessAck -> AXI4 B
  // ---------------------------------------------------------------------------
  // We must collect BOTH AW and W (with W.last) before injecting on TL A.
  // AW and W can arrive in any order on AXI4. We accept each one once per
  // transaction and gate the TL A request on having seen both.
  //
  // FSM:  wIdle  -> gather AW + W into regs
  //       wSendA -> drive PutFullData on TL A, wait for fire
  //       wWaitD -> wait for AccessAck on TL D
  //       wSendB -> drive B response back to AXI master, wait for fire
  val wIdle :: wSendA :: wWaitD :: wSendB :: Nil = Enum(4)
  val wState                                     = RegInit(wIdle)

  val wIdReg   = Reg(UInt(axiParams.idBits.W))
  val wAddrReg = Reg(UInt(axiParams.addrBits.W))
  val wSizeReg = Reg(UInt(axiParams.sizeBits.W))
  val wDataReg = Reg(UInt(axiParams.dataBits.W))
  val wStrbReg = Reg(UInt((axiParams.dataBits / 8).W))

  val wRespReg = Reg(UInt(AXI4Parameters.respBits.W))

  // Track which of AW / W we have already consumed for the current txn.
  val wHaveAw = RegInit(false.B)
  val wHaveW  = RegInit(false.B)

  // Accept AW/W only while idle (or while still gathering this txn's beats).
  val gathering = wState === wIdle
  io.axi.aw.ready := gathering && !wHaveAw
  io.axi.w.ready  := gathering && !wHaveW

  when(io.axi.aw.fire) {
    wIdReg   := io.axi.aw.bits.id
    wAddrReg := io.axi.aw.bits.addr
    wSizeReg := io.axi.aw.bits.size
    wHaveAw  := true.B
    assert(io.axi.aw.bits.len === 0.U, "AXI4ToTLBridge: AW burst length > 0 not supported (got %d)", io.axi.aw.bits.len)
  }
  when(io.axi.w.fire) {
    wDataReg := io.axi.w.bits.data
    wStrbReg := io.axi.w.bits.strb
    wHaveW   := true.B
    assert(io.axi.w.bits.last, "AXI4ToTLBridge: only single-beat W (last=1) is supported")
  }

  // When both AW and W have arrived, advance to send the TL A request.
  val awThisCycle = io.axi.aw.fire
  val wThisCycle  = io.axi.w.fire
  val haveAwNow   = wHaveAw || awThisCycle
  val haveWNow    = wHaveW || wThisCycle
  when(gathering && haveAwNow && haveWNow) {
    wHaveAw := false.B
    wHaveW  := false.B
    wState  := wSendA
  }

  // ---------------------------------------------------------------------------
  // TL A channel arbitration (read priority)
  // ---------------------------------------------------------------------------
  val rWantsA = rState === rSendA
  val wWantsA = wState === wSendA

  // Build the read-A request
  val aReadBits = Wire(io.tl.a.bits.cloneType)
  aReadBits         := DontCare
  aReadBits.opcode  := TLMessages.Get
  aReadBits.param   := 0.U
  aReadBits.size    := rSizeReg
  aReadBits.source  := SRC_READ
  aReadBits.address := rAddrReg
  aReadBits.mask    := MaskGen(rAddrReg, rSizeReg, tlParams.dataBits / 8)
  aReadBits.data    := 0.U
  aReadBits.corrupt := false.B

  // Build the write-A request
  val aWriteBits = Wire(io.tl.a.bits.cloneType)
  aWriteBits         := DontCare
  aWriteBits.opcode  := TLMessages.PutFullData
  aWriteBits.param   := 0.U
  aWriteBits.size    := wSizeReg
  aWriteBits.source  := SRC_WRITE
  aWriteBits.address := wAddrReg
  aWriteBits.mask    := wStrbReg
  aWriteBits.data    := wDataReg
  aWriteBits.corrupt := false.B

  io.tl.a.valid := rWantsA || wWantsA
  io.tl.a.bits  := Mux(rWantsA, aReadBits, aWriteBits)

  // Advance the FSMs on A.fire
  val aReadFire  = io.tl.a.fire && rWantsA
  val aWriteFire = io.tl.a.fire && !rWantsA && wWantsA

  when(aReadFire)(rState  := rWaitD)
  when(aWriteFire)(wState := wWaitD)

  // ---------------------------------------------------------------------------
  // TL D channel: route response back to R or B based on source bit[0]
  // ---------------------------------------------------------------------------
  // SRC_READ has bit0 = 0, SRC_WRITE has bit0 = 1.
  val dIsRead = !io.tl.d.bits.source(0)

  // Only consume D when the corresponding direction has a free landing slot
  // (i.e. its FSM is in WaitD and not already holding a response).
  io.tl.d.ready :=
    (dIsRead && (rState === rWaitD)) ||
      (!dIsRead && (wState === wWaitD))

  val dResp = Mux(io.tl.d.bits.denied || io.tl.d.bits.corrupt, AXI4Parameters.RESP_SLVERR, AXI4Parameters.RESP_OKAY)

  when(io.tl.d.fire && dIsRead) {
    rDataReg := io.tl.d.bits.data
    rRespReg := dResp
    rState   := rSendR
  }
  when(io.tl.d.fire && !dIsRead) {
    wRespReg := dResp
    wState   := wSendB
  }

  // ---------------------------------------------------------------------------
  // AXI4 R channel: drive single-beat response
  // ---------------------------------------------------------------------------
  io.axi.r.valid     := (rState === rSendR)
  io.axi.r.bits      := DontCare
  io.axi.r.bits.id   := rIdReg
  io.axi.r.bits.data := rDataReg
  io.axi.r.bits.resp := rRespReg
  io.axi.r.bits.last := true.B

  when(io.axi.r.fire)(rState := rIdle)

  // ---------------------------------------------------------------------------
  // AXI4 B channel: drive write response
  // ---------------------------------------------------------------------------
  io.axi.b.valid     := (wState === wSendB)
  io.axi.b.bits      := DontCare
  io.axi.b.bits.id   := wIdReg
  io.axi.b.bits.resp := wRespReg

  when(io.axi.b.fire)(wState := wIdle)
}

/**
 * Generate the TileLink mask for a Get/Put given an aligned address and a
 * power-of-two transfer size (in bytes = 2^lgSize).
 *
 * This uses the same algorithm as rocket-chip's util.MaskGen to ensure
 * compatibility with TLMonitor's mask validation.
 *
 * Reference: rocket-chip/src/main/scala/util/Misc.scala:197
 */
private object MaskGen {

  def apply(addr: UInt, lgSize: UInt, beatBytes: Int): UInt = {
    require(isPow2(beatBytes))
    val lgBytes = log2Ceil(beatBytes)
    val sizeOH  = UIntToOH(lgSize | 0.U(log2Up(beatBytes).W), log2Up(beatBytes)) | 1.U

    def helper(i: Int): Seq[(Bool, Bool)] = {
      if (i == 0) {
        Seq((lgSize >= lgBytes.U, true.B))
      } else {
        val sub  = helper(i - 1)
        val size = sizeOH(lgBytes - i)
        val bit  = addr(lgBytes - i)
        val nbit = !bit
        Seq.tabulate(1 << i) { j =>
          val (sub_acc, sub_eq) = sub(j / 2)
          val eq                = sub_eq && (if (j % 2 == 1) bit else nbit)
          val acc               = sub_acc || (size && eq)
          (acc, eq)
        }
      }
    }

    if (beatBytes == 1) 1.U
    else Cat(helper(lgBytes).map(_._1).reverse)
  }

}
