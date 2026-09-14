package memcore.memory.bank

import chisel3._
import chisel3.util._

case class BankParams(addressBits: Int, dataBits: Int, tagBits: Int = 8) {
  require(addressBits > 0 && dataBits > 0 && dataBits % 8 == 0 && tagBits > 0)
  val bytes = dataBits / 8
}

class BankRequest(p: BankParams) extends Bundle {
  val addr  = UInt(p.addressBits.W)
  val write = Bool()
  val data  = UInt(p.dataBits.W)
  val mask  = UInt(p.bytes.W)
  val tag   = UInt(p.tagBits.W)
}

class BankResponse(p: BankParams) extends Bundle {
  val data  = UInt(p.dataBits.W)
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

/** One physical SRAM bank. Cache and NoC policy remain outside this module. */
class RootSramBank(p: BankParams, entries: Int) extends Module {
  require(entries >= 2 && isPow2(entries))

  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new BankRequest(p)))
    val response = Decoupled(new BankResponse(p))
  })

  val memory        = SyncReadMem(entries, Vec(p.bytes, UInt(8.W)))
  val request       = Reg(new BankRequest(p))
  val responseValid = RegInit(false.B)
  val readPending   = RegInit(false.B)
  val responseData  = Reg(UInt(p.dataBits.W))
  val readData      = memory.read(io.request.bits.addr(log2Ceil(entries) - 1, 0), io.request.fire && !io.request.bits.write)

  io.request.ready := !responseValid && !readPending
  when(io.request.fire) {
    request := io.request.bits
    when(io.request.bits.write) {
      memory.write(
        io.request.bits.addr(log2Ceil(entries) - 1, 0),
        io.request.bits.data.asTypeOf(Vec(p.bytes, UInt(8.W))),
        io.request.bits.mask.asBools
      )
      responseData  := 0.U
      responseValid := true.B
    }.otherwise {
      readPending := true.B
    }
  }
  when(readPending) {
    responseData  := readData.asUInt
    responseValid := true.B
    readPending   := false.B
  }

  io.response.valid      := responseValid
  io.response.bits.data  := responseData
  io.response.bits.tag   := request.tag
  io.response.bits.error := false.B
  when(io.response.fire) {
    responseValid := false.B
  }
}

object EmitRootSramBank extends App {
  _root_.circt.stage.ChiselStage.emitSystemVerilogFile(
    new RootSramBank(BankParams(addressBits = 16, dataBits = 64), entries = 64),
    args
  )
}
