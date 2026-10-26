package examples.balls.mxquant

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.mxquant.{Params => ParametersProto}
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.balldomain.ball.blink.{BallStatus, BlinkIO, HasBlink}

@instantiable
class MxquantBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(
    mp.bankWidth == 128 && mp.bankMaskLen == 16,
    "MxquantBall requires 128-bit bank rows and byte write masks"
  )
  require(mp.bankEntries >= 8, "MxquantBall requires eight input rows")
  require(ball.inBW == 1 && ball.outBW == 1, "MxquantBall requires one read and one write")

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val idle :: readReq :: readResp :: exponent :: encode :: codeReq :: codeResp :: scaleReq :: scaleResp :: complete :: Nil =
    Enum(10)
  val state                                                                                                                = RegInit(idle)
  val robId                                                                                                                = Reg(UInt(log2Up(fp.rob_entries).W))
  val source                                                                                                               = Reg(UInt(fp.bank_id_len.W))
  val destination                                                                                                          = Reg(UInt(fp.bank_id_len.W))
  val bases                                                                                                                = Reg(Vec(3, UInt(mp.groupIdWidth.W)))
  val count                                                                                                                = Reg(UInt(fp.iter_len.W))
  val group                                                                                                                = RegInit(0.U(fp.iter_len.W))
  val readRow                                                                                                              = RegInit(0.U(3.W))
  val lane                                                                                                                 = RegInit(0.U(5.W))
  val codeRow                                                                                                              = RegInit(0.U(1.W))
  val maximum                                                                                                              = RegInit(0.U(31.W))
  val blockExponent                                                                                                        = Reg(SInt(10.W))
  val scale                                                                                                                = Reg(UInt(8.W))
  val values                                                                                                               = Reg(Vec(32, UInt(32.W)))
  val codes                                                                                                                = Reg(Vec(32, UInt(8.W)))
  val math                                                                                                                 = Instantiate(new Encode)
  math.io.input         := values(lane)
  math.io.blockExponent := blockExponent

  val active = !reset.asBool
  blink.cmdReq.ready        := state === idle && active
  blink.cmdResp.valid       := state === complete && active
  blink.cmdResp.bits.rob_id := robId
  blink.status.idle         := state === idle
  blink.status.running      := state =/= idle && state =/= complete

  val read = blink.bankRead(0)
  read.rob_id           := robId
  read.bank_id          := source
  read.group_id         := bases(0)
  read.io.req.valid     := state === readReq && active
  read.io.req.bits.addr := (group << 3) + readRow
  read.io.resp.ready    := state === readResp && active

  val write = blink.bankWrite(0)
  write.rob_id           := robId
  write.bank_id          := destination
  write.group_id         := bases(2)
  write.io.req.valid     := (state === codeReq || state === scaleReq) && active
  write.io.req.bits.addr := Mux(state === scaleReq, (count >> 4) + (group >> 4), (group << 1) + codeRow)
  val lowCodes  = Cat((15 to 0 by -1).map(codes(_)))
  val highCodes = Cat((31 to 16 by -1).map(codes(_)))
  write.io.req.bits.data := Mux(
    state === scaleReq,
    (scale.pad(128) << (group(3, 0) << 3))(127, 0),
    Mux(codeRow === 0.U, lowCodes, highCodes)
  )
  write.io.req.bits.mask := VecInit((0 until 16).map(i => state =/= scaleReq || group(3, 0) === i.U))
  write.io.resp.ready    := (state === codeResp || state === scaleResp) && active

  switch(state) {
    is(idle) {
      when(blink.cmdReq.fire) {
        val cmd = blink.cmdReq.bits

        assert(cmd.iter > 0.U && cmd.iter(4, 0) === 0.U, "MxquantBall count must be positive and divisible by 32")
        assert(
          cmd.iter <= (mp.bankEntries * 4).U,
          "MxquantBall input exceeds capacity"
        )
        assert(
          cmd.iter +& (cmd.iter >> 5) <= (mp.bankEntries * 16).U,
          "MxquantBall output exceeds capacity"
        )
        assert(cmd.rs2 === 0.U, "MxquantBall reserved fields must be zero")
        assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "MxquantBall operands must occupy one bank")
        assert(cmd.op1_bank =/= cmd.wr_bank, "MxquantBall banks must differ")
        robId       := blink.cmdReq.bits.rob_id
        source      := cmd.op1_bank
        destination := cmd.wr_bank
        bases       := VecInit(cmd.op1_group, cmd.op2_group, cmd.wr_group)
        count       := cmd.iter
        group       := 0.U
        readRow     := 0.U
        maximum     := 0.U
        state       := readReq
      }
    }
    is(readReq)(when(read.io.req.fire)(state := readResp))
    is(readResp) {
      when(read.io.resp.fire) {
        val words = (0 until 4).map(i => read.io.resp.bits.data(32 * i + 31, 32 * i))
        for (i <- 0 until 4) {
          assert(words(i)(30, 23) =/= 255.U, "MxquantBall input must be finite")
          values(Cat(readRow, i.U(2.W))) := words(i)
        }
        maximum := (Seq(maximum) ++ words.map(_(30, 0))).reduce((a, c) => Mux(a > c, a, c))
        when(readRow === 7.U)(state := exponent).otherwise { readRow := readRow + 1.U; state := readReq }
      }
    }
    is(exponent) {
      val raw      = maximum(30, 23).zext - 135.S(10.W)
      val selected = Mux(maximum === 0.U, 0.S(10.W), Mux(raw < -127.S, -127.S(10.W), raw))
      blockExponent := selected
      scale         := (selected + 127.S).asUInt
      lane          := 0.U
      state         := encode
    }
    is(encode) {
      codes(lane) := math.io.code
      when(lane === 31.U) { codeRow := 0.U; state := codeReq }.otherwise(lane := lane + 1.U)
    }
    is(codeReq)(when(write.io.req.fire)(state := codeResp))
    is(codeResp) {
      when(write.io.resp.fire) {
        when(codeRow === 0.U) { codeRow := 1.U; state := codeReq }.otherwise(state := scaleReq)
      }
    }
    is(scaleReq)(when(write.io.req.fire)(state := scaleResp))
    is(scaleResp) {
      when(write.io.resp.fire) {
        when(group === (count >> 5) - 1.U)(state := complete).otherwise {
          group   := group + 1.U
          readRow := 0.U
          maximum := 0.U
          state   := readReq
        }
      }
    }
    is(complete)(when(blink.cmdResp.fire)(state := idle))
  }
}

object MxquantBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new MxquantBall(fp, mp, bp, id)).blink, None)

}
