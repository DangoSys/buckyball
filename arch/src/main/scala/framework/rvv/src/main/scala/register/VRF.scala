package framework.rvv

import chisel3._
import chisel3.util.Valid
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.rvv.configs.RvvParam

@instantiable
class VRF(val p: RvvParam) extends Module {

  @public
  val io = IO(new Bundle {
    val readAddress = Input(Vec(4, UInt(5.W)))
    val readData    = Output(Vec(4, UInt(p.vLen.W)))

    val write = Flipped(Valid(new Bundle {
      val address = UInt(5.W)
      val data    = UInt(p.vLen.W)
      val mask    = UInt(p.vLen.W)
    }))

  })

  val registers = RegInit(VecInit(Seq.fill(32)(0.U(p.vLen.W))))

  for (port <- 0 until 4) {
    io.readData(port) := registers(io.readAddress(port))
  }

  when(io.write.valid) {
    val current = registers(io.write.bits.address)
    registers(io.write.bits.address) :=
      (current & ~io.write.bits.mask) | (io.write.bits.data & io.write.bits.mask)
  }
}
