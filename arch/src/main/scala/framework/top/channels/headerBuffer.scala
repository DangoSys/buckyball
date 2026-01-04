package framework.top.channels

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

class HeaderBufferData(val b: GlobalConfig) extends Bundle {
  val lock    = Bool()
  val rob_id  = UInt(log2Up(b.frontend.rob_entries).W)
  val bank_id = UInt(log2Up(b.memDomain.bankNum).W)
}

@instantiable
class HeaderBuffer(val b: GlobalConfig) extends Module {

  @public
  val io = IO(new Bundle {
    val read = Output(new HeaderBufferData(b))
    val set  = Flipped(Decoupled(new HeaderBufferData(b)))
  })

  val reg = RegInit(0.U.asTypeOf(new HeaderBufferData(b)))

  io.read      := reg
  io.set.ready := !reg.lock || (io.set.bits.lock === false.B)

  when(io.set.fire) {
    reg := io.set.bits
  }
}
