package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

class WriteReq(val b: GlobalConfig) extends Bundle {
  val bank_id     = UInt(log2Up(b.memDomain.bankNum).W)
  val ball_id     = UInt(log2Up(b.ballDomain.ballNum).W)
  val channel_num = UInt(log2Up(b.top.ballMemChannelNum + 1).W)
  val rob_id      = UInt(log2Up(b.frontend.rob_entries).W)
}

class BankWriteProbe(val b: GlobalConfig) extends Bundle {
  val valid   = Bool()
  val bank_id = UInt(log2Up(b.memDomain.bankNum).W)
  val ball_id = UInt(log2Up(b.ballDomain.ballNum).W)
  val rob_id  = UInt(log2Up(b.frontend.rob_entries).W)
}

@instantiable
class WriteReqGen(val b: GlobalConfig) extends Module {
  val ballIdMappings     = b.ballDomain.ballIdMappings
  val numBalls           = b.ballDomain.ballNum
  val totalWriteChannels = ballIdMappings.map(_.outBW).sum
  val numBanks           = b.memDomain.bankNum

  require(
    ballIdMappings.forall(_.outBW > 0),
    "WriteReqGen assumes every ball has at least one write channel (outBW > 0); otherwise robId indexing is invalid."
  )

  @public
  val io = IO(new Bundle {
    val bank_write_i = Input(Vec(totalWriteChannels, new BankWriteProbe(b)))
    val write_req_o  = Decoupled(new WriteReq(b))
  })

  val reqGroupsWithRobId = (0 until numBalls).flatMap { ballId =>
    (0 until numBanks).map { bankId =>
      val ballOffset       = ballIdMappings.take(ballId).map(_.outBW).sum
      val ballOutBW        = ballIdMappings(ballId).outBW
      val matchingChannels = (ballOffset until ballOffset + ballOutBW).toSeq

      val matchConds = matchingChannels.map { ch =>
        io.bank_write_i(ch).valid &&
        io.bank_write_i(ch).ball_id === ballId.U &&
        io.bank_write_i(ch).bank_id === bankId.U
      }

      val hasReq          = matchConds.reduceOption(_ || _).getOrElse(false.B)
      val channelCountRaw = PopCount(matchConds)

      val maxCh        = b.top.ballMemChannelNum.U
      val channelCount = Mux(channelCountRaw > maxCh, maxCh, channelCountRaw)

      val robId = io.bank_write_i(ballOffset).rob_id

      (hasReq, channelCount, bankId.U, ballId.U, robId)
    }
  }

  val arb = Module(new Arbiter(
    new Bundle {
      val channel_num = UInt(log2Up(b.top.ballMemChannelNum + 1).W)
      val bank_id     = UInt(log2Up(numBanks).W)
      val ball_id     = UInt(log2Up(numBalls).W)
      val rob_id      = UInt(log2Up(b.frontend.rob_entries).W)
    },
    reqGroupsWithRobId.length
  ))

  for (i <- reqGroupsWithRobId.indices) {
    arb.io.in(i).valid            := reqGroupsWithRobId(i)._1
    arb.io.in(i).bits.channel_num := reqGroupsWithRobId(i)._2
    arb.io.in(i).bits.bank_id     := reqGroupsWithRobId(i)._3
    arb.io.in(i).bits.ball_id     := reqGroupsWithRobId(i)._4
    arb.io.in(i).bits.rob_id      := reqGroupsWithRobId(i)._5
  }

  io.write_req_o.valid            := arb.io.out.valid
  io.write_req_o.bits.bank_id     := arb.io.out.bits.bank_id
  io.write_req_o.bits.ball_id     := arb.io.out.bits.ball_id
  io.write_req_o.bits.channel_num := arb.io.out.bits.channel_num
  io.write_req_o.bits.rob_id      := arb.io.out.bits.rob_id
  arb.io.out.ready                := io.write_req_o.ready
}
