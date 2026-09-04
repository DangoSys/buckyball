package examples.balls.lut

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.top.GlobalConfig

@instantiable
class LutBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "LutBall")
    .getOrElse(throw new IllegalArgumentException("LutBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "LUT")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("LUT not found in ballISA"))

  require(b.memDomain.bankWidth == 128, "LutBall requires 128-bit bank rows")
  require(b.memDomain.bankEntries >= 16, "LutBall requires sixteen LUT rows")
  require(mapping.inBW == 2, "LutBall requires inBW=2")
  require(mapping.outBW == 1, "LutBall requires outBW=1")
  require((funct >> 4) == 4, "LUT must encode two reads and one write")

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status
  dontTouch(io)

  private val idle :: lutReq :: lutResp :: inputReq :: inputResp :: transform :: writeReq :: writeResp :: complete :: Nil =
    Enum(9)
  private val state                                                                                                       = RegInit(idle)
  private val robId                                                                                                       = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  private val isSub                                                                                                       = RegInit(false.B)
  private val subRobId                                                                                                    = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  private val inputBank                                                                                                   = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val lutBank                                                                                                     = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val outputBank                                                                                                  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  private val iter                                                                                                        = RegInit(0.U(b.frontend.iter_len.W))
  private val lutRow                                                                                                      = RegInit(0.U(4.W))
  private val inputRow                                                                                                    = RegInit(0.U(log2Ceil(b.memDomain.bankEntries).W))
  private val inputData                                                                                                   = Reg(UInt(128.W))
  private val laneGroup                                                                                                   = RegInit(0.U(2.W))
  private val table                                                                                                       = Reg(Vec(256, UInt(8.W)))
  private val outputWord                                                                                                  = Reg(Vec(16, UInt(8.W)))

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
    io.bankRead(port).group_id         := 0.U
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := 0.U
    io.bankRead(port).io.resp.ready    := false.B
  }
  io.bankRead(0).bank_id := inputBank
  io.bankRead(1).bank_id := lutBank

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := outputBank
  io.bankWrite(0).group_id         := 0.U
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := inputRow
  io.bankWrite(0).io.req.bits.data := Cat(outputWord.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  io.subRobReq.valid := false.B
  io.subRobReq.bits  := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits.cmd
        assert(cmd.funct7 === funct.U, "LutBall funct7 must be LUT")
        assert(cmd.rs2 === 0.U, "LutBall reserves rs2")
        assert(cmd.iter > 0.U && cmd.iter <= b.memDomain.bankEntries.U, "LutBall iter must fit in one bank")
        assert(
          cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
          "LutBall operands must each occupy one bank"
        )
        assert(
          cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank &&
            cmd.op2_bank =/= cmd.wr_bank,
          "LutBall banks must be distinct"
        )
        robId      := io.cmdReq.bits.rob_id
        isSub      := io.cmdReq.bits.is_sub
        subRobId   := io.cmdReq.bits.sub_rob_id
        inputBank  := cmd.op1_bank
        lutBank    := cmd.op2_bank
        outputBank := cmd.wr_bank
        iter       := cmd.iter
        lutRow     := 0.U
        inputRow   := 0.U
        laneGroup  := 0.U
        state      := lutReq
      }
    }
    is(lutReq) {
      io.bankRead(1).io.req.valid            := true.B
      io.bankRead(1).io.req.bits.addr        := lutRow
      when(io.bankRead(1).io.req.fire)(state := lutResp)
    }
    is(lutResp) {
      io.bankRead(1).io.resp.ready := true.B
      when(io.bankRead(1).io.resp.fire) {
        for (lane <- 0 until 16)
          table(Cat(lutRow, lane.U(4.W))) := io.bankRead(1).io.resp.bits.data(8 * lane + 7, 8 * lane)
        when(lutRow === 15.U)(state := inputReq)
          .otherwise { lutRow := lutRow + 1.U; state := lutReq }
      }
    }
    is(inputReq) {
      io.bankRead(0).io.req.valid            := true.B
      io.bankRead(0).io.req.bits.addr        := inputRow
      when(io.bankRead(0).io.req.fire)(state := inputResp)
    }
    is(inputResp) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        inputData := io.bankRead(0).io.resp.bits.data
        laneGroup := 0.U
        state     := transform
      }
    }
    is(transform) {
      for (lane <- 0 until 4) {
        val index = Cat(laneGroup, lane.U(2.W))
        outputWord(index) := table((inputData >> (index << 3))(7, 0))
      }
      when(laneGroup === 3.U)(state := writeReq)
        .otherwise(laneGroup := laneGroup + 1.U)
    }
    is(writeReq) {
      io.bankWrite(0).io.req.valid            := true.B
      when(io.bankWrite(0).io.req.fire)(state := writeResp)
    }
    is(writeResp) {
      io.bankWrite(0).io.resp.ready := true.B
      when(io.bankWrite(0).io.resp.fire) {
        when(inputRow === iter - 1.U)(state := complete)
          .otherwise { inputRow := inputRow + 1.U; state := inputReq }
      }
    }
    is(complete) {
      when(io.cmdResp.fire)(state := idle)
    }
  }
}
