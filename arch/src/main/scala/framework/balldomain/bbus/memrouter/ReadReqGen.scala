package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.blink.{BankRead}

class ReadReq(val b: GlobalConfig) extends Bundle {
  val totalReadChannels = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val bank_id           = UInt(log2Up(b.memDomain.bankNum).W)
  val ball_id           = UInt(log2Up(b.ballDomain.ballNum).W)
  val channel_num       = UInt(log2Up(totalReadChannels).W)
}

@instantiable
class ReadReqGen(val b: GlobalConfig) extends Module {
  val ballIdMappings    = b.ballDomain.ballIdMappings
  val numBalls          = b.ballDomain.ballNum
  val totalReadChannels = ballIdMappings.map(_.inBW).sum
  val numBanks          = b.memDomain.bankNum

  @public
  val io = IO(new Bundle {
    val bank_read_i = Vec(totalReadChannels, Flipped(new BankRead(b)))
    val read_req_o  = Decoupled(new ReadReq(b))
  })

  val reqGroups = (0 until numBalls).flatMap { ballId =>
    (0 until numBanks).map { bankId =>
      val ballOffset       = ballIdMappings.take(ballId).map(_.inBW).sum
      val ballInBW         = ballIdMappings(ballId).inBW
      val matchingChannels = (ballOffset until ballOffset + ballInBW).toSeq

      val hasReq = matchingChannels.map(ch =>
        io.bank_read_i(ch).io.req.valid &&
          io.bank_read_i(ch).ball_id === ballId.U &&
          io.bank_read_i(ch).bank_id === bankId.U
      ).reduceOption(_ || _).getOrElse(false.B)

      val channelCount = PopCount(matchingChannels.map(ch =>
        io.bank_read_i(ch).io.req.valid &&
          io.bank_read_i(ch).ball_id === ballId.U &&
          io.bank_read_i(ch).bank_id === bankId.U
      ))

      (hasReq, channelCount, bankId.U, ballId.U)
    }
  }

  val arb = Module(new Arbiter(
    new Bundle {
      val channel_num = UInt(log2Up(totalReadChannels + 1).W)
      val bank_id     = UInt(log2Up(numBanks).W)
      val ball_id     = UInt(log2Up(numBalls).W)
    },
    reqGroups.length
  ))

  for (i <- reqGroups.indices) {
    arb.io.in(i).valid            := reqGroups(i)._1
    arb.io.in(i).bits.channel_num := reqGroups(i)._2
    arb.io.in(i).bits.bank_id     := reqGroups(i)._3
    arb.io.in(i).bits.ball_id     := reqGroups(i)._4
  }

  io.read_req_o.valid            := arb.io.out.valid
  io.read_req_o.bits.bank_id     := arb.io.out.bits.bank_id
  io.read_req_o.bits.ball_id     := arb.io.out.bits.ball_id
  io.read_req_o.bits.channel_num := arb.io.out.bits.channel_num
  arb.io.out.ready               := io.read_req_o.ready
}
