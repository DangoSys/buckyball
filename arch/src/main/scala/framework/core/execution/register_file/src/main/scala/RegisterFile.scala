package framework.core.execution.register_file

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.core.execution.register_file.configs.{Params => ModuleParam}

class RegisterFileIO(b: ModuleParam) extends Bundle {
  val p = b

  val readAddress = Input(Vec(2, UInt(5.W)))
  val readData    = Output(Vec(2, UInt(p.xLen.W)))

  val write = Flipped(Valid(new Bundle {
    val address = UInt(5.W)
    val data    = UInt(p.xLen.W)
  }))

}

@instantiable
class RegisterFile(b: ModuleParam) extends Module {
  val p = b

  @public
  val io = IO(new RegisterFileIO(b))

  val registers = Reg(Vec(32, UInt(p.xLen.W)))
  when(io.write.valid && io.write.bits.address =/= 0.U) {
    registers(io.write.bits.address) := io.write.bits.data
  }
  for (port <- 0 until 2) {
    val address = io.readAddress(port)
    io.readData(port) := Mux(
      address === 0.U,
      0.U,
      Mux(io.write.valid && io.write.bits.address === address, io.write.bits.data, registers(address))
    )
  }
}
