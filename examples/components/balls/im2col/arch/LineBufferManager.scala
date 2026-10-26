package examples.balls.im2col

import framework.acc.balldomain.configs.BallRouteParam

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.memdomain.BankRead
import examples.balls.im2col.configs.Im2colBallParam

@instantiable
class LineBufferManager(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam,
  val ballParam:     Im2colBallParam)
    extends Module {
  val ballCfg      = ballParam
  val maxIter      = ballCfg.maxIter
  val maxKSize     = ballCfg.maxKSize
  val elemWidth    = ballCfg.inputWidth
  val bankWidth    = memParam.bankWidth
  val lanesPerBeat = bankWidth / elemWidth
  require(isPow2(lanesPerBeat), "LineBuffer lanesPerBeat must be power of 2")
  val maxBeats     = maxIter * maxIter
  require(isPow2(maxBeats) && maxBeats >= 2, "LineBuffer depth must be an even power of two")
  val halfBeats    = maxBeats / 2
  // FIXME: This is a local line-buffer address, sized for maxIter^2; it must not use
  // the external bank address width. Each command's external footprint is
  // checked separately against bankEntries in Im2colConfigRegs.
  val memoryAddrW  = log2Ceil(maxBeats)
  val kW           = log2Ceil(maxKSize + 1)
  val addrW        = log2Ceil(memParam.bankEntries)

  val map = instance

  val inBW = map.inBW

  @public
  val io = IO(new Bundle {

    val bankRead = Vec(
      inBW,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val start     = Input(Bool())
    val inRows    = Input(UInt(16.W))
    val inCols    = Input(UInt(16.W))
    val rowStride = Input(UInt(8.W))
    val colStride = Input(UInt(8.W))
    val padding   = Input(UInt(8.W))
    val startRow  = Input(UInt(8.W))
    val startCol  = Input(UInt(8.W))
    val inputBase = Input(UInt(6.W))
    val lane      = Input(UInt(log2Ceil(lanesPerBeat).W))
    val outRow    = Input(UInt(16.W))
    val outCol    = Input(UInt(16.W))
    val kRowIdx   = Input(UInt(kW.W))
    val kColIdx   = Input(UInt(kW.W))
    val rBankId   = Input(UInt(frontendParam.bank_id_len.W))
    val rGroup    = Input(UInt(memParam.groupIdWidth.W))
    val robId     = Input(UInt(log2Up(frontendParam.rob_entries).W))
    val loadDone  = Output(Bool())
    val elemData  = Output(UInt(elemWidth.W))
  })

  val bufLo    = SyncReadMem(halfBeats, UInt(bankWidth.W))
  val bufHi    = SyncReadMem(halfBeats, UInt(bankWidth.W))
  val active   = RegInit(false.B)
  val pending  = RegInit(false.B)
  val beat     = RegInit(0.U(addrW.W))
  val dimW     = log2Ceil(maxIter + 1)
  val beatsReg = RegInit(0.U((addrW + 1).W))

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := io.rBankId
    io.bankRead(i).rob_id           := io.robId
    io.bankRead(i).group_id         := io.rGroup
  }

  io.bankRead(0).io.req.valid     := active && !pending
  io.bankRead(0).io.req.bits.addr := (io.inputBase +& beat).asTypeOf(UInt(addrW.W))
  io.bankRead(0).io.resp.ready    := pending
  io.loadDone                     := !active

  when(io.start) {
    val rows = io.inRows(dimW - 1, 0)
    val cols = io.inCols(dimW - 1, 0)
    beatsReg := rows * cols
    active   := true.B
    pending  := false.B
    beat     := 0.U
  }.elsewhen(io.bankRead(0).io.req.fire) {
    pending := true.B
  }

  when(io.bankRead(0).io.resp.fire) {
    pending := false.B
    when(beat +& 1.U === beatsReg) {
      active := false.B
    }.otherwise {
      beat := beat + 1.U
    }
  }

  val paddedRow   = io.startRow +& (io.outRow(dimW - 1, 0) * io.rowStride) +& io.kRowIdx
  val paddedCol   = io.startCol +& (io.outCol(dimW - 1, 0) * io.colStride) +& io.kColIdx
  val rowValid    =
    paddedRow >= io.padding && paddedRow < io.padding + io.inRows
  val colValid    =
    paddedCol >= io.padding && paddedCol < io.padding + io.inCols
  val inBound     = rowValid && colValid
  val srcRow      = Mux(inBound, paddedRow - io.padding, 0.U)
  val srcCol      = Mux(inBound, paddedCol - io.padding, 0.U)
  val elemIndex   = srcRow(dimW - 1, 0) * io.inCols(dimW - 1, 0) + srcCol(dimW - 1, 0)
  val beatIndex   = elemIndex
  val laneIndex   = io.lane
  val memoryWrite = io.bankRead(0).io.resp.fire
  val memoryRead  = !active

  val memoryAddress = Mux(
    memoryWrite,
    beat.pad(memoryAddrW),
    beatIndex(memoryAddrW - 1, 0)
  )

  val memoryBank   = memoryAddress(memoryAddrW - 1)
  val memoryRow    = memoryAddress(log2Ceil(halfBeats) - 1, 0)
  val memoryEnable = memoryWrite || memoryRead

  val loData = bufLo.readWrite(
    memoryRow,
    io.bankRead(0).io.resp.bits.data.asUInt,
    memoryEnable && !memoryBank,
    memoryWrite
  )

  val hiData = bufHi.readWrite(
    memoryRow,
    io.bankRead(0).io.resp.bits.data.asUInt,
    memoryEnable && memoryBank,
    memoryWrite
  )

  val readBank   = RegEnable(memoryBank, false.B, memoryRead)
  val memoryData = Mux(readBank, hiData, loData)
  val laneR      = RegInit(0.U(log2Ceil(lanesPerBeat).W))
  val boundR     = RegInit(false.B)
  val beatData   = RegInit(0.U(bankWidth.W))
  val laneD      = RegInit(0.U(log2Ceil(lanesPerBeat).W))
  val boundD     = RegInit(false.B)
  val elemHold   = RegInit(0.U(elemWidth.W))
  laneR    := laneIndex
  boundR   := inBound
  beatData := memoryData
  laneD    := laneR
  boundD   := boundR
  val lanes =
    beatData.asTypeOf(Vec(lanesPerBeat, UInt(elemWidth.W)))
  elemHold := Mux(boundD, lanes(laneD), 0.U)

  io.elemData := elemHold
}
