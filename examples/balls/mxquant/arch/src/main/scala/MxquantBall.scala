package examples.balls.mxquant

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class MxquantBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {
  private val mapping = b.ballDomain.ballIdMappings.find(_.ballName == "MxquantBall").get
  private val funct   = b.ballDomain.ballISA.find(_.mnemonic == "MXQUANT").get.funct7
  require(
    b.memDomain.bankWidth == 128 && b.memDomain.bankMaskLen == 16,
    "MxquantBall requires 128-bit bank rows and byte write masks"
  )
  require(b.memDomain.bankEntries >= 8, "MxquantBall requires eight input rows")
  require(mapping.inBW == 1 && mapping.outBW == 1, "MxquantBall requires one read and one write")
  require((funct >> 4) == 3, "MxquantBall funct7 must encode one read and one write")

  @public
  val io = IO(new BlinkIO(b, 1, 1))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  private val idle :: readReq :: readResp :: exponent :: encode :: codeReq :: codeResp :: scaleReq :: scaleResp :: complete :: Nil =
    Enum(10)
  private val state                                                                                                                = RegInit(idle)
  private val robId                                                                                                                = Reg(UInt(log2Up(b.frontend.rob_entries).W))
  private val subId                                                                                                                = Reg(UInt(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val isSub                                                                                                                = Reg(Bool())
  private val source                                                                                                               = Reg(UInt(b.memDomain.vbankIdWidth.W))
  private val destination                                                                                                          = Reg(UInt(b.memDomain.vbankIdWidth.W))
  private val count                                                                                                                = Reg(UInt(b.frontend.iter_len.W))
  private val group                                                                                                                = RegInit(0.U(b.frontend.iter_len.W))
  private val readRow                                                                                                              = RegInit(0.U(3.W))
  private val lane                                                                                                                 = RegInit(0.U(5.W))
  private val codeRow                                                                                                              = RegInit(0.U(1.W))
  private val maximum                                                                                                              = RegInit(0.U(31.W))
  private val blockExponent                                                                                                        = Reg(SInt(10.W))
  private val scale                                                                                                                = Reg(UInt(8.W))
  private val values                                                                                                               = Reg(Vec(32, UInt(32.W)))
  private val codes                                                                                                                = Reg(Vec(32, UInt(8.W)))
  private val math                                                                                                                 = Module(new Encode)
  math.io.input         := values(lane)
  math.io.blockExponent := blockExponent

  private val active = !reset.asBool
  io.cmdReq.ready            := state === idle && active
  io.cmdResp.valid           := state === complete && active
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subId
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete
  io.subRobReq.valid         := false.B
  io.subRobReq.bits          := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  val read = io.bankRead(0)
  read.rob_id           := robId
  read.ball_id          := 0.U
  read.bank_id          := source
  read.group_id         := 0.U
  read.io.req.valid     := state === readReq && active
  read.io.req.bits.addr := (group << 3) + readRow
  read.io.resp.ready    := state === readResp && active

  val write = io.bankWrite(0)
  write.rob_id           := robId
  write.ball_id          := 0.U
  write.bank_id          := destination
  write.group_id         := 0.U
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
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        assert(cmd.funct7 === funct.U, "MxquantBall unknown funct7")
        assert(cmd.iter > 0.U && cmd.iter(4, 0) === 0.U, "MxquantBall count must be positive and divisible by 32")
        assert(cmd.iter <= (b.memDomain.bankEntries * 4).U, "MxquantBall input exceeds capacity")
        assert(cmd.iter +& (cmd.iter >> 5) <= (b.memDomain.bankEntries * 16).U, "MxquantBall output exceeds capacity")
        assert(cmd.rs1(19, 10) === 0.U && cmd.rs2 === 0.U, "MxquantBall reserved fields must be zero")
        assert(
          cmd.rs1(9, 0) < b.memDomain.virtualBankCount.U && cmd.rs1(29, 20) < b.memDomain.virtualBankCount.U,
          "MxquantBall invalid bank"
        )
        assert(cmd.op1_col === 1.U && cmd.wr_col === 1.U, "MxquantBall operands must occupy one bank")
        assert(cmd.op1_bank =/= cmd.wr_bank, "MxquantBall banks must differ")
        robId       := io.cmdReq.bits.rob_id
        subId       := io.cmdReq.bits.sub_rob_id
        isSub       := io.cmdReq.bits.is_sub
        source      := cmd.op1_bank
        destination := cmd.wr_bank
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
    is(complete)(when(io.cmdResp.fire)(state := idle))
  }
}
