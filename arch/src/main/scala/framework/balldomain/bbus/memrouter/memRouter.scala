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

  // Initialize all output ports to default values
  // TODO: Implement actual routing logic
  for (i <- 0 until bbusChannel) {
    // BankRead outputs
    io.bankRead_o(i).io.req.valid     := false.B
    io.bankRead_o(i).io.req.bits.addr := 0.U
    io.bankRead_o(i).io.resp.ready    := false.B
    io.bankRead_o(i).rob_id           := 0.U
    io.bankRead_o(i).bank_id          := 0.U

    // BankWrite outputs
    io.bankWrite_o(i).io.req.valid      := false.B
    io.bankWrite_o(i).io.req.bits.addr  := 0.U
    io.bankWrite_o(i).io.req.bits.data  := 0.U
    io.bankWrite_o(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite_o(i).io.req.bits.wmode := false.B
    io.bankWrite_o(i).io.resp.ready     := false.B
    io.bankWrite_o(i).rob_id            := 0.U
    io.bankWrite_o(i).bank_id           := 0.U
  }

  // Initialize input ready signals
  for (i <- 0 until numBalls) {
    for (j <- 0 until b.memDomain.bankNum) {
      io.bankRead_i(i)(j).io.req.ready      := false.B
      io.bankRead_i(i)(j).io.resp.valid     := false.B
      io.bankRead_i(i)(j).io.resp.bits.data := 0.U

      io.bankWrite_i(i)(j).io.req.ready    := false.B
      io.bankWrite_i(i)(j).io.resp.valid   := false.B
      io.bankWrite_i(i)(j).io.resp.bits.ok := false.B
    }
  }

  io.bbusConfig_i.ready := false.B

}
