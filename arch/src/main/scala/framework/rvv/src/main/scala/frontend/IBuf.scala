package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.rvv.configs.RvvParam

@instantiable
class IBuf(val p: RvvParam) extends Module {

  @public
  val io = IO(new Bundle {

    val upload = Flipped(Decoupled(new ProgramWrite))

    val uploadEnabled = Input(Bool())
    val fetchEnabled  = Input(Bool())
    val fetchAddress  = Input(UInt(32.W))
    val instruction   = Output(UInt(32.W))
    val loaded        = Output(Bool())
  })

  val memory      = SyncReadMem(p.iBufWords, UInt(32.W))
  val loadedWords = RegInit(0.U((log2Ceil(p.iBufWords) + 1).W))
  val index       = io.fetchAddress(log2Ceil(p.iBufWords) + 1, 2)

  io.upload.ready := io.uploadEnabled
  io.instruction  := memory.read(index, io.fetchEnabled)
  io.loaded       := index < loadedWords

  when(io.upload.fire) {
    assert(io.upload.bits.address < p.iBufWords.U, "instruction upload out of range")
    assert(
      io.upload.bits.address === Mux(io.upload.bits.first, 0.U, loadedWords),
      "instruction upload is not contiguous"
    )
    memory.write(io.upload.bits.address(log2Ceil(p.iBufWords) - 1, 0), io.upload.bits.data)
    loadedWords := io.upload.bits.address + 1.U
  }
}
