package examples.balls.im2col

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.blink.BankRead
import examples.balls.im2col.configs.Im2colBallParam
import framework.top.GlobalConfig

@instantiable
class LineBufferManager(val b: GlobalConfig) extends Module {
  private val ballCfg      = Im2colBallParam(b)
  private val maxIter      = ballCfg.maxIter
  private val maxKSize     = ballCfg.maxKSize
  private val elemWidth    = ballCfg.inputWidth
  private val bankWidth    = b.memDomain.bankWidth
  private val lanesPerBeat = bankWidth / elemWidth
  private val maxBeats     = (maxIter * maxIter + lanesPerBeat - 1) / lanesPerBeat
  private val kW           = log2Ceil(maxKSize + 1)
  private val addrW        = log2Ceil(b.memDomain.bankEntries)

  private val map = b.ballDomain.ballIdMappings
    .find(_.ballName == "Im2colBall")
    .getOrElse(throw new IllegalArgumentException("Im2colBall not found in config"))

  private val inBW = map.inBW

  @public val io = IO(new Bundle {
    val bankRead = Vec(inBW, Flipped(new BankRead(b)))
    val start    = Input(Bool())
    val iter     = Input(UInt(b.frontend.iter_len.W))
    val stride   = Input(UInt(8.W))
    val padding  = Input(UInt(8.W))
    val outRow   = Input(UInt(16.W))
    val outCol   = Input(UInt(16.W))
    val kRowIdx  = Input(UInt(kW.W))
    val kColIdx  = Input(UInt(kW.W))
    val rBankId  = Input(UInt(log2Up(b.memDomain.bankNum).W))
    val robId    = Input(UInt(log2Up(b.frontend.rob_entries).W))
    val loadDone = Output(Bool())
    val elemData = Output(UInt(elemWidth.W))
  })

  private val buf     = RegInit(VecInit(Seq.fill(maxBeats)(0.U(bankWidth.W))))
  private val active  = RegInit(false.B)
  private val pending = RegInit(false.B)
  private val beat    = RegInit(0.U(log2Ceil(maxBeats + 1).W))
  private val totalBeats =
    ((io.iter * io.iter) + (lanesPerBeat - 1).U) / lanesPerBeat.U

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := io.rBankId
    io.bankRead(i).rob_id           := io.robId
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).group_id         := 0.U
  }

  io.bankRead(0).io.req.valid     := active && !pending
  io.bankRead(0).io.req.bits.addr := beat.asTypeOf(UInt(addrW.W))
  io.bankRead(0).io.resp.ready    := pending
  io.loadDone                     := !active

  when(io.start) {
    active  := true.B
    pending := false.B
    beat    := 0.U
  }.elsewhen(io.bankRead(0).io.req.fire) {
    pending := true.B
  }

  when(io.bankRead(0).io.resp.fire) {
    buf(beat) := io.bankRead(0).io.resp.bits.data.asUInt
    pending   := false.B
    when(beat + 1.U === totalBeats) {
      active := false.B
    }.otherwise {
      beat := beat + 1.U
    }
  }

  private val paddedRow = io.outRow * io.stride + io.kRowIdx
  private val paddedCol = io.outCol * io.stride + io.kColIdx
  private val rowValid  = paddedRow >= io.padding && paddedRow < io.padding + io.iter
  private val colValid  = paddedCol >= io.padding && paddedCol < io.padding + io.iter
  private val inBound   = rowValid && colValid
  private val srcRow    = Mux(inBound, paddedRow - io.padding, 0.U)
  private val srcCol    = Mux(inBound, paddedCol - io.padding, 0.U)
  private val elemIndex = srcRow * io.iter + srcCol
  private val beatIndex = elemIndex / lanesPerBeat.U
  private val laneIndex = elemIndex % lanesPerBeat.U
  private val lanes     = buf(beatIndex).asTypeOf(Vec(lanesPerBeat, UInt(elemWidth.W)))

  io.elemData := Mux(inBound, lanes(laneIndex), 0.U)
}
