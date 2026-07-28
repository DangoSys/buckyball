package examples.balls.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.blink.BankWrite
import framework.top.GlobalConfig
import examples.balls.im2col.configs.Im2colBallParam

/**
 * Packs one im2col K tile into a 128-bit bank row.
 *
 * MatrixBall consumes A in (M tile, K tile, tile row) order.  For window w and
 * kernel chunk kt the destination row is:
 *
 *   ((w / 16) * ceil(kernelElems / 16) + kt) * 16 + (w % 16)
 *
 * A partial final K tile is zero-filled to 16 int8 lanes.
 */
@instantiable
class StreamWriter(val b: GlobalConfig) extends Module {
  private val ballCfg      = Im2colBallParam(b)
  private val maxK         = ballCfg.InputNum
  private val elemWidth    = ballCfg.inputWidth
  private val bankWidth    = b.memDomain.bankWidth
  private val lanesPerBeat = bankWidth / elemWidth
  private val laneIdxWidth = log2Ceil(lanesPerBeat)

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))
  private val outBW = mapping.outBW

  require(outBW >= 1, "StreamWriter requires at least one write port")
  require(bankWidth % elemWidth == 0)
  require(lanesPerBeat == 16, "Im2col MatrixBall layout requires 16 int8 lanes")

  @public val io = IO(new Bundle {
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))

    val elemIn    = Flipped(Decoupled(UInt(elemWidth.W)))
    val elemLast  = Input(Bool())
    val init      = Input(Bool())
    val windowIdx = Input(UInt(32.W))
    val kSize     = Input(UInt(log2Ceil(maxK + 1).W))
    val wBankId   = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId     = Input(UInt(log2Up(b.frontend.rob_entries).W))

    val busy           = Output(Bool())
    val windowComplete = Output(Bool())
  })

  private val packCntReg = RegInit(0.U(log2Ceil(lanesPerBeat + 1).W))
  private val packReg =
    RegInit(VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W))))
  private val chunkIdxReg = RegInit(0.U(16.W))
  private val wrPendingReg = RegInit(false.B)
  private val endWindowReg = RegInit(false.B)
  private val wAddrReg = RegInit(0.U(32.W))

  io.busy := wrPendingReg
  io.windowComplete := wrPendingReg && endWindowReg &&
    io.bankWrite(0).io.req.fire

  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask :=
      VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready := false.B
    io.bankWrite(i).bank_id  := 0.U
    io.bankWrite(i).rob_id   := 0.U
    io.bankWrite(i).ball_id  := 0.U
    io.bankWrite(i).group_id := 0.U
  }

  io.bankWrite(0).io.req.valid     := wrPendingReg
  io.bankWrite(0).io.req.bits.addr := wAddrReg
  io.bankWrite(0).io.req.bits.data := Cat(packReg.reverse)
  io.bankWrite(0).io.req.bits.mask :=
    VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready := true.B
  io.bankWrite(0).bank_id  := io.wBankId
  io.bankWrite(0).rob_id   := io.robId
  io.bankWrite(0).group_id := 0.U

  io.elemIn.ready := !wrPendingReg

  val kernelElems = io.kSize * io.kSize
  val kTiles = (kernelElems + (lanesPerBeat - 1).U) >> laneIdxWidth
  val mTile = io.windowIdx >> laneIdxWidth
  val mRow  = io.windowIdx(laneIdxWidth - 1, 0)
  val targetAddr =
    ((mTile * kTiles + chunkIdxReg) << laneIdxWidth) + mRow

  when(io.init) {
    packCntReg   := 0.U
    packReg      := VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W)))
    chunkIdxReg  := 0.U
    wrPendingReg := false.B
    endWindowReg := false.B
    wAddrReg     := 0.U
  }.otherwise {
    when(io.bankWrite(0).io.req.fire) {
      packCntReg   := 0.U
      packReg      := VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W)))
      wrPendingReg := false.B
      endWindowReg := false.B
      when(endWindowReg) {
        chunkIdxReg := 0.U
      }.otherwise {
        chunkIdxReg := chunkIdxReg + 1.U
      }
    }

    when(io.elemIn.fire) {
      packReg(packCntReg(laneIdxWidth - 1, 0)) := io.elemIn.bits
      val nextCnt = packCntReg + 1.U
      packCntReg := nextCnt
      when(nextCnt === lanesPerBeat.U || io.elemLast) {
        wrPendingReg := true.B
        endWindowReg := io.elemLast
        wAddrReg     := targetAddr
      }
    }
  }
}
