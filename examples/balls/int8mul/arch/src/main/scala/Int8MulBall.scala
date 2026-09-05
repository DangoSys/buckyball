package examples.balls.int8mul

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import hardfloat.{recFNFromFN, INToRecFN, MulRecFN, RecFNToIN}
import hardfloat.consts.{round_near_even, tininess_afterRounding}
import freechips.rocketchip.tile.MulAddRecFNPipe

import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class Int8MulBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "Int8MulBall")
    .getOrElse(throw new IllegalArgumentException("Int8MulBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "INT8MUL")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("INT8MUL not found in ballISA"))

  require(mapping.inBW == 2, "Int8MulBall requires inBW=2")
  require(mapping.outBW == 1, "Int8MulBall requires outBW=1")
  require(b.memDomain.bankWidth == 128, "Int8MulBall requires 128-bit bank rows")
  require(b.memDomain.bankEntries <= 64, "Int8MulBall gate_row uses six bits")
  require((funct >> 4) == 4, "INT8MUL must encode two reads and one write")

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val idle :: waitForChannels :: gateRequest :: gateResponse :: inputRequest :: inputResponse :: calculate :: calculateDrain :: writeRequest :: writeResponse :: complete :: Nil =
    Enum(11)
  private val state                                                                                                                                                                      = RegInit(idle)
  private val robId                                                                                                                                                                      = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub                                                                                                                                                                      = RegInit(false.B)
  private val subRobId                                                                                                                                                                   = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val gateBank                                                                                                                                                                   = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val inputBank                                                                                                                                                                  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank                                                                                                                                                                 = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val groups                                                                                                                                                                     = RegInit(0.U(5.W))
  private val group                                                                                                                                                                      = RegInit(0.U(5.W))
  private val iter                                                                                                                                                                       = RegInit(0.U(b.frontend.iter_len.W))
  private val gateRow                                                                                                                                                                    = RegInit(0.U(6.W))
  private val inputRow                                                                                                                                                                   = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val ratio                                                                                                                                                                      = Reg(UInt(32.W))
  private val gateWord                                                                                                                                                                   = Reg(UInt(128.W))
  private val inputWord                                                                                                                                                                  = Reg(UInt(128.W))
  private val lane                                                                                                                                                                       = RegInit(0.U(4.W))
  private val outputWord                                                                                                                                                                 = Reg(Vec(16, UInt(8.W)))

  private val productReg      = Reg(UInt(16.W))
  private val productLaneReg  = Reg(UInt(4.W))
  private val productValid    = RegInit(false.B)
  private val productFloatReg = Reg(UInt(33.W))
  private val floatLaneReg    = Reg(UInt(4.W))
  private val floatValid      = RegInit(false.B)
  private val scaledReg       = Reg(UInt(33.W))
  private val scaledLaneReg   = Reg(UInt(4.W))
  private val scaledValid     = RegInit(false.B)

  private def positiveFinite(value: UInt): Bool =
    !value(31) && value(30, 23) =/= 255.U && value(30, 0) =/= 0.U

  private val gateValue      = (gateWord >> (lane << 3))(7, 0).asSInt
  private val inputValue     = (inputWord >> (lane << 3))(7, 0).asSInt
  private val integerProduct = (gateValue * inputValue).asUInt
  private val productToFloat = Module(new INToRecFN(16, 8, 24))
  private val scale          = Module(new MulAddRecFNPipe(3, 8, 24))
  private val toInt8         = Module(new RecFNToIN(8, 24, 8))

  productToFloat.io.signedIn       := true.B
  productToFloat.io.in             := productReg
  productToFloat.io.roundingMode   := round_near_even
  productToFloat.io.detectTininess := tininess_afterRounding
  scale.io.validin                 := floatValid
  scale.io.op                      := 0.U
  scale.io.a                       := productFloatReg
  scale.io.b                       := recFNFromFN(8, 24, ratio)
  scale.io.c                       := recFNFromFN(8, 24, 0.U)
  scale.io.roundingMode            := round_near_even
  scale.io.detectTininess          := tininess_afterRounding.asUInt
  toInt8.io.in                     := scaledReg
  toInt8.io.roundingMode           := round_near_even
  toInt8.io.signedOut              := true.B

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
    io.bankRead(port).bank_id          := Mux(port.U === 0.U, gateBank, inputBank)
    io.bankRead(port).group_id         := group
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := Mux(port.U === 0.U, gateRow, inputRow)
    io.bankRead(port).io.resp.ready    := false.B
  }

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := group
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := inputRow
  io.bankWrite(0).io.req.bits.data := Cat(outputWord.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        assert(cmd.funct7 === funct.U, "Int8MulBall funct7 must be INT8MUL")
        assert(cmd.rs2(63, 38) === 0.U, "Int8MulBall reserves rs2 bits 63:38")
        assert(positiveFinite(cmd.rs2(31, 0)), "Int8MulBall ratio must be finite and positive")
        assert(cmd.rs2(37, 32) < b.memDomain.bankEntries.U, "Int8MulBall gate_row must fit one physical bank")
        assert(
          cmd.op1_col =/= 0.U && cmd.op1_col === cmd.op2_col && cmd.op1_col === cmd.wr_col,
          "Int8MulBall bank groups must match"
        )
        assert(cmd.op1_col <= b.memDomain.bankNum.U, "Int8MulBall bank groups exceed physical banks")
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
          "Int8MulBall banks must be distinct"
        )
        assert(cmd.iter =/= 0.U && cmd.iter <= b.memDomain.bankEntries.U, "Int8MulBall iter must fit one physical bank")
        robId      := io.cmdReq.bits.rob_id
        isSub      := io.cmdReq.bits.is_sub
        subRobId   := io.cmdReq.bits.sub_rob_id
        gateBank   := cmd.op1_bank
        inputBank  := cmd.op2_bank
        outputBank := cmd.wr_bank
        groups     := cmd.op1_col
        group      := 0.U
        iter       := cmd.iter
        gateRow    := cmd.rs2(37, 32)
        inputRow   := 0.U
        ratio      := cmd.rs2(31, 0)
        lane       := 0.U
        state      := waitForChannels
      }
    }
    is(waitForChannels) {
      when(io.channelReady)(state := gateRequest)
    }
    is(gateRequest) {
      io.bankRead(0).io.req.valid            := true.B
      when(io.bankRead(0).io.req.fire)(state := gateResponse)
    }
    is(gateResponse) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        gateWord := io.bankRead(0).io.resp.bits.data
        state    := inputRequest
      }
    }
    is(inputRequest) {
      io.bankRead(1).io.req.valid            := true.B
      when(io.bankRead(1).io.req.fire)(state := inputResponse)
    }
    is(inputResponse) {
      io.bankRead(1).io.resp.ready := true.B
      when(io.bankRead(1).io.resp.fire) {
        inputWord := io.bankRead(1).io.resp.bits.data
        lane      := 0.U
        state     := calculate
      }
    }
    is(calculate) {
      productReg        := integerProduct
      productLaneReg    := lane
      productValid      := true.B
      when(lane === 15.U)(state := calculateDrain)
        .otherwise(lane := lane + 1.U)
    }
    is(calculateDrain) {
      when(scaledValid && scaledLaneReg === 15.U)(state := writeRequest)
    }
    is(writeRequest) {
      io.bankWrite(0).io.req.valid            := true.B
      when(io.bankWrite(0).io.req.fire)(state := writeResponse)
    }
    is(writeResponse) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(inputRow +& 1.U === iter) {
          inputRow := 0.U
          when(group +& 1.U === groups)(state := complete)
            .otherwise { group := group + 1.U; state := gateRequest }
        }.otherwise { inputRow := inputRow + 1.U; state := inputRequest }
      }
    }
    is(complete) {
      when(io.cmdResp.fire)(state := idle)
    }
  }

  when(productValid) {
    productFloatReg := productToFloat.io.out
    floatLaneReg    := productLaneReg
  }
  when(scale.io.validout) {
    scaledReg     := scale.io.out
    scaledLaneReg := Pipe(floatValid, floatLaneReg, 3).bits
  }
  when(scaledValid) {
    outputWord(scaledLaneReg) := toInt8.io.out
  }

  productValid := state === calculate
  floatValid   := productValid
  scaledValid  := scale.io.validout
}
