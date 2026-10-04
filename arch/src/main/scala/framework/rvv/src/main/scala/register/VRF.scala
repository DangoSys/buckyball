package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class VRF(val b: GlobalConfig) extends Module {
  private val p                = b.rvv
  private val wordBits         = 64
  private val wordsPerRegister = p.vLen / wordBits

  @public val io = IO(new Bundle {
    val initialize = Input(Bool())
    val readWords  = Output(Vec(32, Vec(wordsPerRegister, UInt(wordBits.W))))

    val write = Vec(
      p.laneNumber,
      Flipped(Valid(new Bundle {
        val address = UInt(5.W)
        val word    = UInt(log2Ceil(wordsPerRegister).W)
        val data    = UInt(wordBits.W)
        val mask    = UInt(wordBits.W)
      }))
    )

  })

  val registers = RegInit(
    VecInit(Seq.fill(32)(VecInit(Seq.fill(wordsPerRegister)(0.U(wordBits.W)))))
  )

  io.readWords := registers
  for {
    register <- 0 until 32
    word     <- 0 until wordsPerRegister
  } {
    val selected = io.write.map(w => w.valid && w.bits.address === register.U && w.bits.word === word.U)
    val mask     = io.write
      .zip(selected)
      .map { case (w, enable) => Mux(enable, w.bits.mask, 0.U) }
      .reduce(_ | _)
    val data     = io.write
      .zip(selected)
      .map { case (w, enable) => Mux(enable, w.bits.data & w.bits.mask, 0.U) }
      .reduce(_ | _)
    when(io.initialize) {
      registers(register)(word) := 0.U
    }.elsewhen(mask.orR) {
      registers(register)(word) := (registers(register)(word) & ~mask) | data
    }
  }
}
