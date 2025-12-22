package examples.toy.balldomain.emptyball

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import examples.toy.balldomain.BallDomainParam
import framework.balldomain.blink.{BallRegist, Blink}

@instantiable
class EmptyBall(
  parameter:   BallDomainParam,
  id:          Int,
  bankEntries: Int,
  bankWidth:   Int,
  bankMaskLen: Int)
    extends Module
    with BallRegist {
  @public
  val io     = IO(new Blink(parameter, bankEntries, bankWidth, bankMaskLen))
  val ballId = id.U

  def Blink: Blink = io

  io.cmdResp.valid       := RegNext(io.cmdReq.valid)
  io.cmdResp.bits.rob_id := RegNext(io.cmdReq.bits.rob_id)
  io.cmdReq.ready        := true.B

  for (i <- 0 until parameter.numBanks) {
    io.bankRead(i).io.req.valid        := false.B
    io.bankRead(i).io.req.bits.addr    := 0.U
    io.bankRead(i).io.req.bits.fromDMA := false.B
    io.bankRead(i).io.resp.ready       := false.B
    io.bankRead(i).rob_id              := 0.U
    io.bankRead(i).bank_id             := 0.U

    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).rob_id            := 0.U
    io.bankWrite(i).bank_id           := 0.U
    io.bankWrite(i).io.req.bits.wmode := false.B
  }
  io.status.ready := true.B
  io.status.valid    := io.cmdResp.valid
  io.status.idle     := false.B
  io.status.init     := false.B
  io.status.running  := false.B
  io.status.iter     := 0.U
  io.status.complete := io.cmdResp.valid

}
