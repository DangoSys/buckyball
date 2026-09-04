package examples.balls.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.blink.BankWrite
import framework.top.GlobalConfig
import examples.balls.im2col.configs.Im2colBallParam

/**
 * Packs im2col K tiles into 128-bit bank rows (SMatMulBall A layout).
 *
 *   ((w / 16) * ceil(kernelElems / 16) + kt) * 16 + (w % 16)
 *
 * Partial final K-tile lanes and partial final M-tile rows are zero-filled.
 */
@instantiable
class StreamWriter(val b: GlobalConfig) extends Module {
  private val ballCfg      = Im2colBallParam(b)
  private val maxKSize     = ballCfg.maxKSize
  private val elemWidth    = ballCfg.inputWidth
  private val bankWidth    = b.memDomain.bankWidth
  private val bankEntries  = b.memDomain.bankEntries
  private val lanesPerBeat = bankWidth / elemWidth
  private val laneIdxWidth = log2Ceil(lanesPerBeat)
  private val kW           = log2Ceil(maxKSize + 1)
  private val bankAddrW    = log2Ceil(bankEntries)
  private val maxKTiles    =
    (maxKSize * maxKSize + lanesPerBeat - 1) / lanesPerBeat
  private val kTilesW      = log2Ceil(maxKTiles + 1)
  require(
    isPow2(bankEntries),
    "StreamWriter requires power-of-two bank depth"
  )

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(
      throw new IllegalArgumentException("Im2colBall not found in config")
    )

  private val outBW = mapping.outBW

  require(outBW == 1, "StreamWriter requires outBW=1")
  require(bankWidth % elemWidth == 0)
  require(lanesPerBeat == 16, "Im2col SMatMulBall layout requires 16 int8 lanes")

  @public val io = IO(new Bundle {
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))

    val elemIn      = Flipped(Decoupled(UInt(elemWidth.W)))
    val elemLast    = Input(Bool())
    val finalWindow = Input(Bool())
    val init        = Input(Bool())
    val windowIdx   = Input(UInt(32.W))
    val kRows       = Input(UInt(kW.W))
    val kCols       = Input(UInt(kW.W))
    val wBankId     = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId       = Input(UInt(log2Up(b.frontend.rob_entries).W))

    val busy           = Output(Bool())
    val windowComplete = Output(Bool())
    val opComplete     = Output(Bool())
  })

  private val zeros = VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W)))

  private val packCntReg = RegInit(0.U(log2Ceil(lanesPerBeat + 1).W))

  private val packReg = RegInit(
    VecInit(Seq.fill(lanesPerBeat)(0.U(elemWidth.W)))
  )

  private val chunkIdxReg  = RegInit(0.U(kTilesW.W))
  private val wrPendingReg = RegInit(false.B)
  private val issued       = RegInit(false.B)
  private val endWindowReg = RegInit(false.B)
  private val finalWinReg  = RegInit(false.B)
  private val padActive    = RegInit(false.B)
  private val padMTile     = RegInit(0.U(32.W))
  private val padKTiles    = RegInit(0.U(kTilesW.W))
  private val padK         = RegInit(0.U(kTilesW.W))
  private val padRow       = RegInit(0.U(laneIdxWidth.W))
  private val padRow0      = RegInit(0.U(laneIdxWidth.W))
  private val wAddrReg     = RegInit(0.U(bankAddrW.W))

  private val kernelElems = io.kRows * io.kCols
  private val kTilesReg   = RegInit(0.U(kTilesW.W))

  private val mTile = io.windowIdx >> laneIdxWidth
  private val mRow  = io.windowIdx(laneIdxWidth - 1, 0)

  private def tileOff(tile: UInt, kTiles: UInt): UInt = {
    MuxLookup(kTiles, 0.U)(
      Seq(
        1.U -> tile,
        2.U -> (tile << 1),
        3.U -> ((tile << 1) + tile),
        4.U -> (tile << 2)
      )
    )
  }

  private val targetAddr =
    ((tileOff(mTile, kTilesReg) + chunkIdxReg) << laneIdxWidth) + mRow
  private val padAddr    =
    ((tileOff(padMTile, padKTiles) + padK) << laneIdxWidth) + padRow
  private val writeFire  = io.bankWrite(0).io.req.fire
  private val writeResp  = io.bankWrite(0).io.resp.fire

  io.busy           := wrPendingReg || padActive
  io.windowComplete := wrPendingReg && endWindowReg && writeResp && !finalWinReg
  io.opComplete     := Mux(
    padActive,
    wrPendingReg && writeResp &&
      (padK + 1.U === padKTiles) && (padRow === (lanesPerBeat - 1).U),
    wrPendingReg && endWindowReg && writeResp && finalWinReg &&
      (mRow === (lanesPerBeat - 1).U)
  )

  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask :=
      VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready    := false.B
    io.bankWrite(i).bank_id          := 0.U
    io.bankWrite(i).rob_id           := 0.U
    io.bankWrite(i).ball_id          := 0.U
    io.bankWrite(i).group_id         := 0.U
  }

  io.bankWrite(0).io.req.valid     := wrPendingReg && !issued
  io.bankWrite(0).io.req.bits.addr := wAddrReg
  io.bankWrite(0).io.req.bits.data := Cat(packReg.reverse)
  io.bankWrite(0).io.req.bits.mask :=
    VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := true.B
  io.bankWrite(0).bank_id          := io.wBankId
  io.bankWrite(0).rob_id           := io.robId
  io.bankWrite(0).group_id         := 0.U

  io.elemIn.ready := !wrPendingReg && !padActive

  when(io.init) {
    packCntReg   := 0.U
    packReg      := zeros
    chunkIdxReg  := 0.U
    wrPendingReg := false.B
    issued       := false.B
    endWindowReg := false.B
    finalWinReg  := false.B
    wAddrReg     := 0.U
    padActive    := false.B
    padMTile     := 0.U
    padKTiles    := 0.U
    padK         := 0.U
    padRow       := 0.U
    padRow0      := 0.U
    kTilesReg    := ((kernelElems +& (lanesPerBeat - 1).U) >> laneIdxWidth)
      .asTypeOf(UInt(kTilesW.W))
    assert(
      ((kernelElems +& (lanesPerBeat - 1).U) >> laneIdxWidth) >= 1.U &&
        ((kernelElems +& (lanesPerBeat - 1).U) >> laneIdxWidth) <= 4.U,
      "StreamWriter kTiles must be 1..4"
    )
  }.otherwise {
    when(writeFire) {
      issued := true.B
    }
    when(writeResp) {
      packCntReg   := 0.U
      packReg      := zeros
      wrPendingReg := false.B
      issued       := false.B
      endWindowReg := false.B
      when(padActive) {
        when(padRow === (lanesPerBeat - 1).U) {
          when(padK + 1.U === padKTiles) {
            padActive := false.B
            padK      := 0.U
            padRow    := 0.U
          }.otherwise {
            padK   := padK + 1.U
            padRow := padRow0
          }
        }.otherwise {
          padRow := padRow + 1.U
        }
      }.elsewhen(endWindowReg) {
        chunkIdxReg := 0.U
        when(finalWinReg && (mRow =/= (lanesPerBeat - 1).U)) {
          padActive := true.B
          padMTile  := mTile
          padKTiles := kTilesReg
          padK      := 0.U
          padRow    := mRow + 1.U
          padRow0   := mRow + 1.U
        }
        finalWinReg := false.B
      }.otherwise {
        chunkIdxReg := chunkIdxReg + 1.U
      }
    }

    when(padActive && !wrPendingReg) {
      wrPendingReg := true.B
      endWindowReg := false.B
      finalWinReg  := false.B
      packReg      := zeros
      wAddrReg     := padAddr.asTypeOf(UInt(bankAddrW.W))
    }

    when(io.elemIn.fire) {
      packReg(packCntReg(laneIdxWidth - 1, 0)) := io.elemIn.bits
      val nextCnt = packCntReg + 1.U
      packCntReg := nextCnt
      when(nextCnt === lanesPerBeat.U || io.elemLast) {
        wrPendingReg := true.B
        endWindowReg := io.elemLast
        finalWinReg  := io.elemLast && io.finalWindow
        wAddrReg     := targetAddr.asTypeOf(UInt(bankAddrW.W))
      }
    }
  }
}
