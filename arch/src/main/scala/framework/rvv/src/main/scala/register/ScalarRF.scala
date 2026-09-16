package framework.rvv

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}

@instantiable
class ScalarRF extends Module {

  @public
  val io = IO(new Bundle {
    val initialize = Flipped(Valid(Vec(8, UInt(32.W))))

    val xReadAddress1 = Input(UInt(5.W))
    val xReadAddress2 = Input(UInt(5.W))
    val xReadData1    = Output(UInt(32.W))
    val xReadData2    = Output(UInt(32.W))

    val xWrite = Flipped(Valid(new Bundle {
      val address = UInt(5.W)
      val data    = UInt(32.W)
    }))

    val fReadAddress1 = Input(UInt(5.W))
    val fReadAddress2 = Input(UInt(5.W))
    val fReadAddress3 = Input(UInt(5.W))
    val fReadData1    = Output(UInt(32.W))
    val fReadData2    = Output(UInt(32.W))
    val fReadData3    = Output(UInt(32.W))

    val fWrite = Flipped(Valid(new Bundle {
      val address = UInt(5.W)
      val data    = UInt(32.W)
    }))

  })

  val x = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))
  val f = RegInit(VecInit(Seq.fill(32)(0.U(32.W))))

  io.xReadData1 := x(io.xReadAddress1)
  io.xReadData2 := x(io.xReadAddress2)
  io.fReadData1 := f(io.fReadAddress1)
  io.fReadData2 := f(io.fReadAddress2)
  io.fReadData3 := f(io.fReadAddress3)

  when(io.initialize.valid) {
    x.foreach(_ := 0.U)
    f.foreach(_ := 0.U)
    for (i <- 0 until 8) {
      x(10 + i) := io.initialize.bits(i)
    }
  }.otherwise {
    when(io.xWrite.valid && io.xWrite.bits.address =/= 0.U) {
      x(io.xWrite.bits.address) := io.xWrite.bits.data
    }
    when(io.fWrite.valid) {
      f(io.fWrite.bits.address) := io.fWrite.bits.data
    }
  }
}
