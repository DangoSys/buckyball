package examples.balls.int8add

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import hardfloat.{recFNFromFN, AddRecFN, INToRecFN, MulRecFN, RecFNToIN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}

import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class Int8AddBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Int8AddBall")
    .getOrElse(throw new IllegalArgumentException("Int8AddBall not found in config"))

  private val addFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "INT8ADD")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("INT8ADD not found in ballISA"))

  private val reluFunct = b.ballDomain.ballISA
    .find(_.mnemonic == "INT8ADD_RELU")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("INT8ADD_RELU not found in ballISA"))

  require(mapping.inBW == 2, "Int8AddBall requires inBW=2")
  require(mapping.outBW == 1, "Int8AddBall requires outBW=1")
  require(b.memDomain.bankWidth == 128, "Int8AddBall requires 128-bit bank rows")
  require((addFunct >> 4) == 4 && (reluFunct >> 4) == 4, "Int8AddBall instructions must encode two reads and one write")

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val idle :: waitForChannels :: readRequest :: readResponse :: writeRequest :: writeResponse :: complete :: Nil =
    Enum(7)
  private val state                                                                                                      = RegInit(idle)
  private val robId                                                                                                      = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub                                                                                                      = RegInit(false.B)
  private val subRobId                                                                                                   = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val lhsBank                                                                                                    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val rhsBank                                                                                                    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank                                                                                                 = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val groups                                                                                                     = RegInit(0.U(5.W))
  private val group                                                                                                      = RegInit(0.U(5.W))
  private val iter                                                                                                       = RegInit(0.U(b.frontend.iter_len.W))
  private val row                                                                                                        = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val lhsRatio                                                                                                   = Reg(UInt(32.W))
  private val rhsRatio                                                                                                   = Reg(UInt(32.W))
  private val relu                                                                                                       = RegInit(false.B)
  private val lhsWord                                                                                                    = Reg(UInt(128.W))
  private val rhsWord                                                                                                    = Reg(UInt(128.W))
  private val lhsRequested                                                                                               = RegInit(false.B)
  private val rhsRequested                                                                                               = RegInit(false.B)
  private val lhsReceived                                                                                                = RegInit(false.B)
  private val rhsReceived                                                                                                = RegInit(false.B)

  private def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  private val lhsToFloat = Seq.fill(16)(Module(new INToRecFN(8, 8, 24)))
  private val rhsToFloat = Seq.fill(16)(Module(new INToRecFN(8, 8, 24)))
  private val lhsMul     = Seq.fill(16)(Module(new MulRecFN(8, 24)))
  private val rhsMul     = Seq.fill(16)(Module(new MulRecFN(8, 24)))
  private val add        = Seq.fill(16)(Module(new AddRecFN(8, 24)))
  private val toInt      = Seq.fill(16)(Module(new RecFNToIN(8, 24, 8)))
  private val result     = Wire(Vec(16, UInt(8.W)))

  for (lane <- 0 until 16) {
    lhsToFloat(lane).io.signedIn       := true.B
    lhsToFloat(lane).io.in             := lhsWord(8 * lane + 7, 8 * lane)
    lhsToFloat(lane).io.roundingMode   := round_near_even
    lhsToFloat(lane).io.detectTininess := tininess_afterRounding
    rhsToFloat(lane).io.signedIn       := true.B
    rhsToFloat(lane).io.in             := rhsWord(8 * lane + 7, 8 * lane)
    rhsToFloat(lane).io.roundingMode   := round_near_even
    rhsToFloat(lane).io.detectTininess := tininess_afterRounding

    lhsMul(lane).io.a              := lhsToFloat(lane).io.out
    lhsMul(lane).io.b              := recFNFromFN(8, 24, lhsRatio)
    lhsMul(lane).io.roundingMode   := round_near_even
    lhsMul(lane).io.detectTininess := tininess_afterRounding.asBool
    rhsMul(lane).io.a              := rhsToFloat(lane).io.out
    rhsMul(lane).io.b              := recFNFromFN(8, 24, rhsRatio)
    rhsMul(lane).io.roundingMode   := round_near_even
    rhsMul(lane).io.detectTininess := tininess_afterRounding.asBool

    add(lane).io.subOp          := false.B
    add(lane).io.a              := lhsMul(lane).io.out
    add(lane).io.b              := rhsMul(lane).io.out
    add(lane).io.roundingMode   := round_near_even
    add(lane).io.detectTininess := tininess_afterRounding.asBool
    toInt(lane).io.in           := add(lane).io.out
    toInt(lane).io.roundingMode := round_near_even
    toInt(lane).io.signedOut    := true.B
    result(lane)                := Mux(relu && toInt(lane).io.out(7), 0.U, toInt(lane).io.out)
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId
  io.status.idle             := state === idle
  io.status.running          := state =/= idle && state =/= complete

  for (port <- 0 until 2) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).ball_id          := 0.U
    io.bankRead(port).bank_id          := Mux(port.U === 0.U, lhsBank, rhsBank)
    io.bankRead(port).group_id         := group
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := row
    io.bankRead(port).io.resp.ready    := false.B
  }
  io.bankWrite(0).rob_id := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := group
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := row
  io.bankWrite(0).io.req.bits.data := Cat(result.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B
  io.subRobReq.valid               := false.B
  io.subRobReq.bits                := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        assert(cmd.funct7 === addFunct.U || cmd.funct7 === reluFunct.U, "Int8AddBall received an unknown funct7")
        assert(
          positiveFinite(cmd.rs2(31, 0)) && positiveFinite(cmd.rs2(63, 32)),
          "Int8AddBall ratios must be finite and positive"
        )
        assert(
          cmd.op1_col =/= 0.U && cmd.op1_col === cmd.op2_col && cmd.op1_col === cmd.wr_col,
          "Int8AddBall bank groups must match"
        )
        assert(cmd.op1_col <= b.memDomain.bankNum.U, "Int8AddBall bank groups exceed physical banks")
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "Int8AddBall banks must be distinct"
        )
        assert(cmd.iter =/= 0.U && cmd.iter <= b.memDomain.bankEntries.U, "Int8AddBall iter must fit one physical bank")
        robId        := io.cmdReq.bits.rob_id
        isSub        := io.cmdReq.bits.is_sub
        subRobId     := io.cmdReq.bits.sub_rob_id
        lhsBank      := cmd.op1_bank
        rhsBank      := cmd.op2_bank
        outputBank   := cmd.wr_bank
        groups       := cmd.op1_col
        group        := 0.U
        iter         := cmd.iter
        row          := 0.U
        lhsRatio     := cmd.rs2(31, 0)
        rhsRatio     := cmd.rs2(63, 32)
        relu         := cmd.funct7 === reluFunct.U
        lhsRequested := false.B
        rhsRequested := false.B
        lhsReceived  := false.B
        rhsReceived  := false.B
        state        := waitForChannels
      }
    }
    is(waitForChannels) {
      when(io.channelReady)(state := readRequest)
    }
    is(readRequest) {
      io.bankRead(0).io.req.valid                   := !lhsRequested
      io.bankRead(1).io.req.valid                   := !rhsRequested
      when(io.bankRead(0).io.req.fire)(lhsRequested := true.B)
      when(io.bankRead(1).io.req.fire)(rhsRequested := true.B)
      when((lhsRequested || io.bankRead(0).io.req.fire) && (rhsRequested || io.bankRead(1).io.req.fire)) {
        state := readResponse
      }
    }
    is(readResponse) {
      io.bankRead(0).io.resp.ready := !lhsReceived
      io.bankRead(1).io.resp.ready := !rhsReceived
      when(io.bankRead(0).io.resp.fire) { lhsWord := io.bankRead(0).io.resp.bits.data; lhsReceived := true.B }
      when(io.bankRead(1).io.resp.fire) { rhsWord := io.bankRead(1).io.resp.bits.data; rhsReceived := true.B }
      when((lhsReceived || io.bankRead(0).io.resp.fire) && (rhsReceived || io.bankRead(1).io.resp.fire)) {
        state := writeRequest
      }
    }
    is(writeRequest) {
      io.bankWrite(0).io.req.valid            := true.B
      when(io.bankWrite(0).io.req.fire)(state := writeResponse)
    }
    is(writeResponse) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        lhsRequested := false.B
        rhsRequested := false.B
        lhsReceived  := false.B
        rhsReceived  := false.B
        when(row +& 1.U === iter) {
          row := 0.U
          when(group +& 1.U === groups)(state := complete)
            .otherwise { group := group + 1.U; state := readRequest }
        }.otherwise { row := row + 1.U; state := readRequest }
      }
    }
    is(complete) {
      when(io.cmdResp.fire)(state := idle)
    }
  }
}
