package examples.balls.f32add

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instantiate}
import framework.balldomain.blink.{BallStatus, BlinkIO, HasBallStatus, HasBlink, SubRobRow}
import framework.balldomain.blink.mmio.{MmioRead, MmioWrite}
import framework.balldomain.rs.BallRsIssue
import framework.top.GlobalConfig

@instantiable
class F32AddBall(val b: GlobalConfig) extends Module with HasBlink with HasBallStatus {
  private val mapping = b.ballDomain.ballIdMappings.find(_.ballName == "F32AddBall")
    .getOrElse(throw new IllegalArgumentException("F32AddBall not found in config"))
  private val funct   = b.ballDomain.ballISA.find(_.mnemonic == "F32ADD").map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("F32ADD not found in ballISA"))
  require(mapping.inBW == 2 && mapping.outBW == 1)
  require(funct == 76)
  require(b.memDomain.bankWidth == 128 && b.memDomain.bankMaskLen == 16)

  @public val io = IO(new BlinkIO(b, mapping.inBW, mapping.outBW))
  def blink:  BlinkIO    = io
  def status: BallStatus = io.status

  private val idle :: channels :: reading :: calculate :: writing :: acknowledge :: complete :: Nil = Enum(7)
  private val state                                                                                 = RegInit(idle)
  private val issue                                                                                 = Reg(new BallRsIssue(b))
  private val row                                                                                   = RegInit(0.U(16.W))
  private val requested                                                                             = RegInit(VecInit(Seq.fill(2)(false.B)))
  private val received                                                                              = RegInit(VecInit(Seq.fill(2)(false.B)))
  private val words                                                                                 = Reg(Vec(2, UInt(128.W)))
  private val output                                                                                = Reg(UInt(128.W))
  private val first                                                                                 = issue.cmd.rs2(5)
  private val sourceGroup                                                                           = issue.cmd.rs2(4, 0)
  private val add                                                                                   = Instantiate(new F32AddRow)
  add.io.source      := words(0)
  add.io.accumulator := Mux(first, 0.U(128.W), words(1))

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := issue.rob_id
  io.cmdResp.bits.is_sub     := issue.is_sub
  io.cmdResp.bits.sub_rob_id := issue.sub_rob_id
  io.status.idle             := state === idle
  io.status.running          := state =/= idle
  io.subRobReq.valid         := false.B
  io.subRobReq.bits          := SubRobRow.tieOff(b)
  MmioRead.tieOff(io.mmioRead)
  MmioWrite.tieOff(io.mmioWrite)

  for (port <- 0 until 2) {
    val read   = io.bankRead(port)
    val active = if (port == 0) true.B else !first
    read.rob_id                            := issue.rob_id
    read.ball_id                           := 0.U
    read.bank_id                           := (if (port == 0) issue.cmd.op1_bank else issue.cmd.op2_bank)
    read.group_id                          := (if (port == 0) sourceGroup else 0.U)
    read.io.req.valid                      := state === reading && active && !requested(port)
    read.io.req.bits.addr                  := row
    read.io.resp.ready                     := state === reading && active && requested(port) && !received(port)
    when(read.io.req.fire)(requested(port) := true.B)
    when(read.io.resp.fire) {
      words(port)    := read.io.resp.bits.data
      received(port) := true.B
    }
  }

  private val write = io.bankWrite(0)
  write.rob_id           := issue.rob_id
  write.ball_id          := 0.U
  write.bank_id          := issue.cmd.wr_bank
  write.group_id         := 0.U
  write.io.req.valid     := state === writing
  write.io.req.bits.addr := row
  write.io.req.bits.data := output
  write.io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  write.io.resp.ready    := state === acknowledge

  when(io.cmdReq.fire) {
    val cmd = io.cmdReq.bits.cmd
    assert(cmd.funct7 === funct.U, "F32ADD invalid funct7")
    assert(cmd.op1_en && cmd.op2_en && cmd.wr_spad_en, "F32ADD requires two inputs and one output")
    assert(cmd.rs2(63, 6) === 0.U, "F32ADD reserved rs2 bits")
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
      "F32ADD banks must differ"
    )
    assert(
      cmd.rs2(4, 0) < cmd.op1_col && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
      "F32ADD source group must exist and accumulators must have one group"
    )
    for (bank <- Seq(cmd.op1_bank, cmd.op2_bank, cmd.wr_bank)) {
      assert(bank < b.memDomain.virtualBankCount.U, "F32ADD invalid virtual bank")
      val shared = bank >= b.frontend.shared_bank_id_base.U
      assert(!shared || b.memDomain.sharedEnable.B, "F32ADD shared memory is disabled")
      val depth  = Mux(shared, b.memDomain.sharedBankEntries.U, b.memDomain.bankEntries.U)
      assert(cmd.iter =/= 0.U && cmd.iter <= depth, "F32ADD row span exceeds bank")
    }
    issue := io.cmdReq.bits
    row                 := 0.U
    requested.foreach(_ := false.B)
    received.foreach(_  := false.B)
    state               := channels
  }
  when(state === channels && io.channelReady)(state                      := reading)
  when(state === reading && received(0) && (first || received(1)))(state := calculate)
  when(state === calculate) {
    output := add.io.sum
    state  := writing
  }
  when(write.io.req.fire)(state                                          := acknowledge)
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "F32ADD bank write failed")
    when(row +& 1.U === issue.cmd.iter) {
      state := complete
    }.otherwise {
      row                 := row + 1.U
      requested.foreach(_ := false.B)
      received.foreach(_  := false.B)
      state               := reading
    }
  }
  when(io.cmdResp.fire)(state                                            := idle)
}
