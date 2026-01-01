package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import framework.balldomain.blink.{BankRead, BankWrite}
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

@instantiable
class MemRouter(val b: GlobalConfig) extends Module {
  val bbusChannel = b.ballDomain.bbusChannel
  val numBalls    = b.ballDomain.ballNum

  @public
  val io = IO(new Bundle {
    val bankRead_i  = Vec(numBalls, Vec(b.memDomain.bankNum, new BankRead(b)))
    val bankWrite_i = Vec(numBalls, Vec(b.memDomain.bankNum, new BankWrite(b)))

    val bankRead_o  = Vec(bbusChannel, Flipped(new BankRead(b)))
    val bankWrite_o = Vec(bbusChannel, Flipped(new BankWrite(b)))
  })

  for (i <- 0 until bbusChannel) {
    io.bankRead_o(i).io.req.valid     := false.B
    io.bankRead_o(i).io.req.bits.addr := 0.U
    io.bankRead_o(i).io.resp.ready    := false.B
    io.bankRead_o(i).rob_id           := 0.U
    io.bankRead_o(i).bank_id          := 0.U

    io.bankWrite_o(i).io.req.valid      := false.B
    io.bankWrite_o(i).io.req.bits.addr  := 0.U
    io.bankWrite_o(i).io.req.bits.data  := 0.U
    io.bankWrite_o(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite_o(i).io.req.bits.wmode := false.B
    io.bankWrite_o(i).io.resp.ready     := false.B
    io.bankWrite_o(i).rob_id            := 0.U
    io.bankWrite_o(i).bank_id           := 0.U
  }

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

}
