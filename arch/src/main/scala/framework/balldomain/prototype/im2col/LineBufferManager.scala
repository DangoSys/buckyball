package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.blink.BankRead
import framework.balldomain.prototype.im2col.configs.Im2colBallParam
import framework.top.GlobalConfig

@instantiable
class LineBufferManager(val b: GlobalConfig) extends Module {
  private val maxK          = Im2colBallParam().InputNum
  private val elemWidth     = Im2colBallParam().inputWidth
  private val bankWidth     = b.memDomain.bankWidth
  private val lanesPerBeat  = 16
  private val maxInColWords = 3

  private val map = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))

  private val inBW = map.inBW

  @public val io = IO(new Bundle {
    val bankRead      = Vec(inBW, Flipped(new BankRead(b)))
    val startPreload  = Input(Bool())
    val startLoadNext = Input(Bool())
    val kRow          = Input(UInt(log2Ceil(maxK + 1).W))
    val inCol         = Input(UInt(16.W))
    val rowPtr        = Input(UInt(16.W))
    val rBaseBeat     = Input(UInt(32.W))
    val rBankId       = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId         = Input(UInt(log2Up(b.frontend.rob_entries).W))
    val loadDone      = Output(Bool())

    val elemReq = new Bundle {
      val kRowIdx = Input(UInt(log2Ceil(maxK + 1).W))
      val kColIdx = Input(UInt(log2Ceil(maxK + 1).W))
      val colPtr  = Input(UInt(16.W))
    }

    val elemData = Output(UInt(elemWidth.W))
  })

  private def ceilDiv(a: UInt, d: Int): UInt = (a + (d - 1).U) / d.U
  private val inColWords = ceilDiv(io.inCol + (lanesPerBeat - 1).U, lanesPerBeat)
  private val buf        = RegInit(VecInit(Seq.fill(maxK)(VecInit(Seq.fill(maxInColWords)(0.U(bankWidth.W))))))

  private val rowFifo = Module(new RowSlotFIFO(maxK))
  private val loader  = Module(new LineLoadCtrl(maxK, maxInColWords))
  rowFifo.io.kRows   := io.kRow
  rowFifo.io.init    := io.startPreload
  rowFifo.io.advance := loader.io.advanceFifo

  loader.io.startPreload  := io.startPreload
  loader.io.startLoadNext := io.startLoadNext
  loader.io.kRow          := io.kRow
  loader.io.inColWords    := inColWords
  loader.io.inCol         := io.inCol
  loader.io.rowPtr        := io.rowPtr
  loader.io.rBaseBeat     := io.rBaseBeat
  loader.io.targetSlot    := rowFifo.io.slotToOverwrite
  loader.io.reqReady      := io.bankRead(0).io.req.ready
  loader.io.respValid     := io.bankRead(0).io.resp.valid

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := io.rBankId
    io.bankRead(i).rob_id           := io.robId
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).group_id         := 0.U
  }

  io.bankRead(0).io.req.valid     := loader.io.reqValid
  io.bankRead(0).io.req.bits.addr := loader.io.reqAddr
  io.bankRead(0).io.resp.ready    := loader.io.respReady
  io.loadDone                     := loader.io.done

  when(io.bankRead(0).io.resp.fire) {
    buf(loader.io.writeRow)(loader.io.writeBeat) := io.bankRead(0).io.resp.bits.data.asUInt
  }

  private val rowByte   = (io.rowPtr + io.elemReq.kRowIdx) * io.inCol
  private val startLane = ((rowByte + io.elemReq.colPtr) % lanesPerBeat.U)(log2Ceil(lanesPerBeat) - 1, 0)
  private val slot      = RowSlotFIFO
    .logicalToPhysical(rowFifo.io.head, io.elemReq.kRowIdx, io.kRow)(log2Ceil(maxK) - 1, 0)
  private val laneSum   = startLane + io.elemReq.kColIdx
  private val beatIdx   = (laneSum / lanesPerBeat.U)(log2Ceil(maxInColWords) - 1, 0)
  private val laneIdx   = (laneSum % lanesPerBeat.U)(log2Ceil(lanesPerBeat) - 1, 0)
  private val lanes     = buf(slot)(beatIdx).asTypeOf(Vec(lanesPerBeat, UInt(elemWidth.W)))
  io.elemData := lanes(laneIdx)
}
