package framework.rvv

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.rvv.configs.RvvParam

@instantiable
class DBuf(val p: RvvParam) extends Module {

  @public
  val io = IO(new Bundle {
    val load        = Flipped(Decoupled(new DataWrite))
    val loadEnabled = Input(Bool())
    val request     = Vec(p.memoryPorts, Flipped(Decoupled(new VectorMemoryRequest)))
    val response    = Vec(p.memoryPorts, Decoupled(new VectorMemoryResponse))
  })

  val memory        = Seq.fill(4)(SyncReadMem(p.dBufWords, UInt(8.W)))
  val pending       = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val pendingWrite  = Reg(Vec(p.memoryPorts, Bool()))
  val pendingError  = Reg(Vec(p.memoryPorts, Bool()))
  val responseValid = RegInit(VecInit(Seq.fill(p.memoryPorts)(false.B)))
  val responseData  = Reg(Vec(p.memoryPorts, UInt(32.W)))
  val responseError = Reg(Vec(p.memoryPorts, Bool()))

  io.load.ready := io.loadEnabled
  when(io.load.fire) {
    assert(io.load.bits.address < p.dBufWords.U, "DBUF load address out of range")
    for (byte <- 0 until 4) {
      when(io.load.bits.mask(byte)) {
        memory(byte).write(
          io.load.bits.address(log2Ceil(p.dBufWords) - 1, 0),
          io.load.bits.data(8 * byte + 7, 8 * byte)
        )
      }
    }
  }

  for (port <- 0 until p.memoryPorts) {
    val aligned  = !io.request(port).bits.address(1, 0).orR
    val inRange  = (io.request(port).bits.address >> 2) < p.dBufWords.U
    val readData = Cat((0 until 4).reverse.map { byte =>
      memory(byte).read(
        io.request(port).bits.address(log2Ceil(p.dBufWords) + 1, 2),
        io.request(port).fire && !io.request(port).bits.write && aligned && inRange
      )
    })

    io.request(port).ready       := !pending(port) && !responseValid(port)
    io.response(port).valid      := responseValid(port)
    io.response(port).bits.data  := responseData(port)
    io.response(port).bits.error := responseError(port)

    when(io.request(port).fire) {
      pending(port)      := true.B
      pendingWrite(port) := io.request(port).bits.write
      pendingError(port) := !aligned || !inRange
      when(io.request(port).bits.write && aligned && inRange) {
        for (byte <- 0 until 4) {
          memory(byte).write(
            io.request(port).bits.address(log2Ceil(p.dBufWords) + 1, 2),
            io.request(port).bits.data(8 * byte + 7, 8 * byte)
          )
        }
      }
    }

    when(pending(port)) {
      pending(port)       := false.B
      responseValid(port) := true.B
      responseData(port)  := Mux(pendingWrite(port), 0.U, readData)
      responseError(port) := pendingError(port)
    }

    when(io.response(port).fire) {
      responseValid(port) := false.B
    }
  }
}
