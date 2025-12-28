package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import framework.balldomain.bbus.BBusConfigIO
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig

@instantiable
class MemRouter(val b: GlobalConfig, val numBalls: Int, val bbusChannel: Int) extends Module {

  @public
  val io = IO(new Bundle {

    val bankRead_i  = Vec(numBalls, Vec(b.memDomain.bankNum, new BankRead(b)))
    val bankWrite_i = Vec(numBalls, Vec(b.memDomain.bankNum, new BankWrite(b)))

    val bbusConfig_i = Flipped(Decoupled(new BBusConfigIO(numBalls)))

    // Output: bbusChannel channels to MemDomain frontend
    val bankRead_o  = Vec(bbusChannel, Flipped(new BankRead(b)))
    val bankWrite_o = Vec(bbusChannel, Flipped(new BankWrite(b)))

  })

}
