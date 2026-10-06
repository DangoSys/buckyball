package memcore.memory.bank

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

class BankRequest(p: BankSetParams) extends Bundle {
  val addr  = UInt(p.rowBits.W)
  val write = Bool()
  val data  = UInt(p.dataBits.W)
  val mask  = UInt(p.bytes.W)
  val tag   = UInt(p.tagBits.W)
}

class BankResponse(p: BankSetParams) extends Bundle {
  val data  = UInt(p.dataBits.W)
  val tag   = UInt(p.tagBits.W)
  val error = Bool()
}

/** One physical SRAM bank. Cache and NoC policy remain outside this module. */
@instantiable
class Bank(p: BankSetParams) extends Module {

  @public
  val io = IO(new Bundle {
    val request  = Flipped(Decoupled(new BankRequest(p)))
    val response = Decoupled(new BankResponse(p))
  })

  val memory        = SyncReadMem(p.entriesPerBank, Vec(p.bytes, UInt(8.W)))
  val request       = Reg(new BankRequest(p))
  val responseValid = RegInit(false.B)
  val readPending   = RegInit(false.B)
  val responseData  = Reg(UInt(p.dataBits.W))

  val readData = memory.readWrite(
    io.request.bits.addr,
    io.request.bits.data.asTypeOf(Vec(p.bytes, UInt(8.W))),
    io.request.bits.mask.asBools,
    io.request.fire,
    io.request.bits.write
  )

  io.request.ready := !responseValid && !readPending
  when(io.request.fire) {
    request := io.request.bits
    when(io.request.bits.write) {
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
