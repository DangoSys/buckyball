package framework.balldomain.prototype.im2col

import chisel3._
import chisel3.util._

class RowSlotFIFO(maxRows: Int) extends Module {

  val io = IO(new Bundle {
    val kRows           = Input(UInt(log2Ceil(maxRows + 1).W))
    val init            = Input(Bool())
    val advance         = Input(Bool())
    val head            = Output(UInt(log2Ceil(maxRows).W))
    val slotToOverwrite = Output(UInt(log2Ceil(maxRows).W))
  })

  private val headReg = RegInit(0.U(log2Ceil(maxRows).W))

  io.head            := headReg
  io.slotToOverwrite := headReg

  when(io.init) {
    headReg := 0.U
  }.elsewhen(io.advance && (io.kRows > 0.U)) {
    when(headReg + 1.U === io.kRows) {
      headReg := 0.U
    }.otherwise {
      headReg := headReg + 1.U
    }
  }
}

object RowSlotFIFO {

  def logicalToPhysical(head: UInt, logicalRow: UInt, kRows: UInt): UInt = {
    val sum = head + logicalRow
    Mux(sum >= kRows, sum - kRows, sum)
  }

}
