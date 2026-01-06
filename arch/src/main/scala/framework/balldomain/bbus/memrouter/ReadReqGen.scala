package framework.balldomain.bbus.memrouter

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig

class ReadReq(val b: GlobalConfig) extends Bundle {
  val totalReadChannels = b.ballDomain.ballIdMappings.map(_.inBW).sum
  val bank_id           = UInt(log2Up(b.memDomain.bankNum).W)
  val ball_id           = UInt(log2Up(b.ballDomain.ballNum).W)
  // 按你的要求：不加 +1
  val channel_num       = UInt(log2Up(totalReadChannels).W)
  val rob_id            = UInt(log2Up(b.frontend.rob_entries).W)
}

// 统计探针：只带“统计需要的信息”，不带 ready/resp
class BankReadProbe(val b: GlobalConfig) extends Bundle {
  val valid   = Bool()
  val bank_id = UInt(log2Up(b.memDomain.bankNum).W)
  val ball_id = UInt(log2Up(b.ballDomain.ballNum).W)
  val rob_id  = UInt(log2Up(b.frontend.rob_entries).W)
}

@instantiable
class ReadReqGen(val b: GlobalConfig) extends Module {
  val ballIdMappings    = b.ballDomain.ballIdMappings
  val numBalls          = b.ballDomain.ballNum
  val totalReadChannels = ballIdMappings.map(_.inBW).sum
  val numBanks          = b.memDomain.bankNum

  require(
    ballIdMappings.forall(_.inBW > 0),
    "ReadReqGen assumes every ball has at least one read channel (inBW > 0); otherwise robId indexing is invalid."
  )

  @public
  val io = IO(new Bundle {
    val bank_read_i = Input(Vec(totalReadChannels, new BankReadProbe(b)))
    val read_req_o  = Decoupled(new ReadReq(b))
  })

  val reqGroupsWithRobId = (0 until numBalls).flatMap { ballId =>
    (0 until numBanks).map { bankId =>
      val ballOffset       = ballIdMappings.take(ballId).map(_.inBW).sum
      val ballInBW         = ballIdMappings(ballId).inBW
      val matchingChannels = (ballOffset until ballOffset + ballInBW).toSeq

      val matchConds = matchingChannels.map { ch =>
        io.bank_read_i(ch).valid &&
          io.bank_read_i(ch).ball_id === ballId.U &&
          io.bank_read_i(ch).bank_id === bankId.U
      }

      val hasReq       = matchConds.reduceOption(_ || _).getOrElse(false.B)
      val channelCount = PopCount(matchConds)

      val robId = io.bank_read_i(ballOffset).rob_id

      (hasReq, channelCount, bankId.U, ballId.U, robId)
    }
  }

  val arb = Module(new Arbiter(
    new Bundle {
      // 按你的要求：不加 +1，且和 ReadReq 保持一致
      val channel_num = UInt(log2Up(totalReadChannels).W)
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

  io.read_req_o.valid            := arb.io.out.valid
  io.read_req_o.bits.bank_id     := arb.io.out.bits.bank_id
  io.read_req_o.bits.ball_id     := arb.io.out.bits.ball_id
  io.read_req_o.bits.channel_num := arb.io.out.bits.channel_num
  io.read_req_o.bits.rob_id      := arb.io.out.bits.rob_id
  arb.io.out.ready               := io.read_req_o.ready
}
