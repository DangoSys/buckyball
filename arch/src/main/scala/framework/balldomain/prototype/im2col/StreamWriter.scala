package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.blink.BankWrite
import framework.top.GlobalConfig
import framework.balldomain.prototype.im2col.configs.Im2colBallParam

/**
 * StreamWriter — packs elements into beats and writes to SRAM.
 *
 * Accepts one element per cycle via elemIn, packs lanesPerBeat elements
 * into a full beat, then issues a write request. Handles partial flush
 * at window end.
 */
@instantiable
class StreamWriter(val b: GlobalConfig) extends Module {
  private val maxK         = Im2colBallParam().InputNum
  private val elemWidth    = Im2colBallParam().inputWidth
  private val bankWidth    = b.memDomain.bankWidth
  private val lanesPerBeat = 16

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))

  private val outBW = mapping.outBW

  @public val io = IO(new Bundle {
    // SRAM write port
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))

    // Element input
    val elemIn = Flipped(Decoupled(UInt(elemWidth.W)))

    // Control
    val start = Input(Bool()) // pulse: reset pack state for new window (does NOT reset write address)
    val init  = Input(Bool()) // pulse: initialize write address for new operation
    val flush = Input(Bool()) // pulse: flush partial pack at window end

    // Configuration
    val wBaseBeat = Input(UInt(32.W)) // initial write address (used only on init)
    val wBankId   = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId     = Input(UInt(log2Up(b.frontend.rob_entries).W))

    // Status
    val busy         = Output(Bool())     // actively writing or have pending data
    val wBaseBeatOut = Output(UInt(32.W)) // updated write address
  })

  private val packCntReg   = RegInit(0.U(log2Ceil(lanesPerBeat + 1).W))
  private val packReg      = RegInit(VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W))))
  private val wrPendingReg = RegInit(false.B)
  private val wAddrReg     = RegInit(0.U(32.W))
  private val flushingReg  = RegInit(false.B)

  io.wBaseBeatOut := wAddrReg
  io.busy         := wrPendingReg || flushingReg

  // Default bankWrite signals
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
    io.bankWrite(i).bank_id           := io.wBankId
    io.bankWrite(i).rob_id            := io.robId
    io.bankWrite(i).ball_id           := 0.U
    io.bankWrite(i).group_id          := 0.U
  }

  io.bankWrite(0).io.resp.ready := true.B

  // Write request when pack full or flushing
  io.bankWrite(0).io.req.valid      := wrPendingReg
  io.bankWrite(0).io.req.bits.addr  := wAddrReg
  io.bankWrite(0).io.req.bits.data  := Cat(packReg.reverse)
  io.bankWrite(0).io.req.bits.wmode := true.B
  io.bankWrite(0).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))

  // Accept elements when not pending a write
  io.elemIn.ready := !wrPendingReg && !flushingReg

  when(io.init) {
    wAddrReg     := io.wBaseBeat
    packCntReg   := 0.U
    wrPendingReg := false.B
    flushingReg  := false.B
  }

  when(io.start) {
    packCntReg   := 0.U
    wrPendingReg := false.B
    flushingReg  := false.B
  }

  when(io.bankWrite(0).io.req.fire) {
    wAddrReg     := wAddrReg + 1.U
    packCntReg   := 0.U
    wrPendingReg := false.B
    flushingReg  := false.B
  }

  when(io.elemIn.fire) {
    packReg(packCntReg) := io.elemIn.bits
    val nextCnt = packCntReg + 1.U
    packCntReg := nextCnt
    when(nextCnt === lanesPerBeat.U) {
      wrPendingReg := true.B
    }
  }

  when(io.flush && !wrPendingReg) {
    when(packCntReg > 0.U) {
      wrPendingReg := true.B
      flushingReg  := true.B
    }
  }
}
