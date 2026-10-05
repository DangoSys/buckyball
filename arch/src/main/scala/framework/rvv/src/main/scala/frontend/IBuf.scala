package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class IBuf(val b: GlobalConfig) extends Module {
  private val p = b.rvv

  @public
  val io = IO(new Bundle {

    val upload = Flipped(Decoupled(new ProgramWrite))

    val uploadEnabled = Input(Bool())
    val fetchEnabled  = Input(Bool())
    val fetchAddress  = Input(UInt(32.W))
    val instruction   = Output(UInt(32.W))
    val loaded        = Output(Bool())
    val wordCount     = Output(UInt(32.W))
  })

  val memory      = SyncReadMem(p.iBufWords, UInt(32.W))
  val loadedWords = RegInit(0.U((log2Ceil(p.iBufWords) + 1).W))
  val indexWidth  = math.max(1, log2Ceil(p.iBufWords))
  val index       = (io.fetchAddress >> 2)(indexWidth - 1, 0)

  io.upload.ready := io.uploadEnabled && !io.fetchEnabled
  io.instruction  := memory.readWrite(
    Mux(io.upload.fire, io.upload.bits.address(indexWidth - 1, 0), index),
    io.upload.bits.data,
    io.fetchEnabled || io.upload.fire,
    io.upload.fire
  )
  io.loaded       := io.fetchAddress < (p.iBufWords * 4).U && index < loadedWords
  io.wordCount    := loadedWords

  when(io.upload.fire) {
    assert(io.upload.bits.address < p.iBufWords.U, "instruction upload out of range")
    assert(
      io.upload.bits.address === Mux(io.upload.bits.first, 0.U, loadedWords),
      "instruction upload is not contiguous"
    )
    loadedWords := io.upload.bits.address + 1.U
  }
}
