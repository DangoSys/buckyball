package examples.balls.redmule

import framework.acc.balldomain.bbus.{BallFactory, BallPorts}
import buckyball.balls.redmule.{Params => ParametersProto}
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.acc.balldomain.ball.blink.{BallCommand, BallStatus, BlinkIO, HasBlink}

@instantiable
class RedmuleBall(
  fp: FrontendParam,
  mp: MemDomainParam,
  bp: BallDomainParam,
  id: Int)
    extends Module
    with HasBlink {
  val ball = bp.balls.find(_.id == id).get

  ball.parameters.unpack(classOf[ParametersProto])

  require(ball.inBW == 2 && ball.outBW == 1)
  require(
    mp.bankWidth == 128 && mp.bankMaskLen == 16
  )
  require(
    mp.bankEntries <= 65535 && mp.sharedBankEntries <= 65535
  )

  @public
  val blink = IO(new BlinkIO(fp, mp, bp, id))

  val Seq(idle, channels, configure, controlReply, running, complete) = Enum(6)
  val state                                                           = RegInit(idle)

  val issue = Reg(new BallCommand(fp, mp))

  val controlStep = RegInit(0.U(4.W))
  val finished    = RegInit(false.B)
  val rows        = issue.rs2(11, 0)
  val aBase       = issue.rs2(21, 12)
  val bBase       = issue.rs2(31, 22)
  val cBase       = issue.rs2(41, 32)
  val first       = issue.rs2(42)
  val native      = Instantiate(new RedmuleNative)
  native.io.clk_i       := clock
  native.io.rst_ni      := !reset.asBool
  native.io.ctrl_req_i  := state === configure
  native.io.ctrl_wen_i  := controlStep === 0.U
  native.io.ctrl_addr_i := VecInit(Seq(4, 44, 48, 52, 32, 36, 40, 60, 0).map(_.U(32.W)))(controlStep)
  native.io.ctrl_data_i := VecInit(Seq(
    0.U(32.W),
    16777216.U | (aBase << 4),
    33554432.U | (bBase << 4),
    50331648.U | (cBase << 4),
    (8.U(32.W) << 16) | rows,
    41943040.U(32.W) | ((!first).asUInt << 20) | issue.iter(15, 0),
    0.U(32.W),
    0.U(32.W),
    0.U(32.W)
  ))(controlStep)

  val Seq(busIdle, readRequest, readReply, writeRequest, writeReply, advance, grant, response) = Enum(8)
  val busState                                                                                 = RegInit(busIdle)
  val address                                                                                  = Reg(UInt(32.W))
  val load                                                                                     = Reg(Bool())
  val data                                                                                     = Reg(UInt(256.W))
  val mask                                                                                     = Reg(UInt(32.W))
  val transactionId                                                                            = Reg(UInt(8.W))
  val half                                                                                     = RegInit(false.B)
  val responseData                                                                             = Reg(UInt(256.W))
  val region                                                                                   = address(25, 24)
  val bankRow                                                                                  = address(23, 4) +& half
  val startRow                                                                                 = Mux(region === 1.U, aBase, Mux(region === 2.U, bBase, cBase))
  val span                                                                                     = Mux(region === 1.U, rows * (issue.iter(15, 0) >> 3), Mux(region === 2.U, issue.iter(15, 0), rows))
  val inWindow                                                                                 = bankRow >= startRow && bankRow < (startRow +& span)
  val readPort                                                                                 = Mux(region === 2.U, 1.U, 0.U)

  val readBank = Mux(
    region === 1.U,
    issue.op1_bank,
    Mux(region === 2.U, issue.op2_bank, issue.wr_bank)
  )

  val readGroup = Mux(region === 1.U, issue.op1_group, Mux(region === 2.U, issue.op2_group, issue.wr_group))

  val writeMask = Mux(half, mask(31, 16), mask(15, 0))
  val writeData = Mux(half, data(255, 128), data(127, 0))
  native.io.mem_gnt_i   := busState === grant
  native.io.mem_valid_i := busState === response
  native.io.mem_data_i  := responseData
  native.io.mem_id_i    := transactionId

  for (port <- 0 until 2) {
    val read = blink.bankRead(port)
    read.rob_id                     := issue.rob_id
    read.bank_id                    := readBank
    read.group_id                   := readGroup
    read.io.req.valid               := busState === readRequest && readPort === port.U && inWindow
    read.io.req.bits.addr           := bankRow
    read.io.resp.ready              := busState === readReply && readPort === port.U
    when(read.io.req.fire)(busState := readReply)
    when(read.io.resp.fire) {
      responseData := Mux(
        half,
        Cat(read.io.resp.bits.data, responseData(127, 0)),
        Cat(responseData(255, 128), read.io.resp.bits.data)
      )
      busState     := advance
    }
  }
  val write = blink.bankWrite(0)
  write.rob_id                                                              := issue.rob_id
  write.bank_id                                                             := issue.wr_bank
  write.group_id                                                            := issue.wr_group
  write.io.req.valid                                                        := busState === writeRequest && writeMask.orR && inWindow
  write.io.req.bits.addr                                                    := bankRow
  write.io.req.bits.data                                                    := writeData
  write.io.req.bits.mask                                                    := VecInit((0 until 16).map(writeMask(_)))
  write.io.resp.ready                                                       := busState === writeReply
  when(write.io.req.fire) {
    assert(region === 3.U && inWindow, "RedMulE store outside C window")
    busState := writeReply
  }
  when(write.io.resp.fire) {
    assert(write.io.resp.bits.ok, "RedMulE bank write failed")
    busState := advance
  }
  when(busState === busIdle && state === running && native.io.mem_req_o) {
    assert(
      native.io.mem_addr_o(31, 26) === 0.U && native.io.mem_addr_o(25, 24) =/= 0.U &&
        native.io.mem_addr_o(3, 0) === 0.U,
      "RedMulE address cannot map to a bank row"
    )
    address       := native.io.mem_addr_o
    data          := native.io.mem_data_o
    mask          := native.io.mem_be_o
    load          := native.io.mem_wen_o
    transactionId := native.io.mem_id_o
    half          := false.B
    responseData  := 0.U
    busState      := Mux(native.io.mem_wen_o, readRequest, writeRequest)
  }
  // Unused lanes in the native 256-bit load are zero-filled at a tensor boundary.
  when(busState === readRequest && !inWindow)(busState                      := advance)
  when(busState === writeRequest && (!writeMask.orR || !inWindow))(busState := advance)
  when(busState === advance) {
    when(!half) {
      half     := true.B
      busState := Mux(load, readRequest, writeRequest)
    }.otherwise(busState := grant)
  }
  when(busState === grant) {
    assert(
      native.io.mem_req_o && native.io.mem_addr_o === address && native.io.mem_wen_o === load,
      "RedMulE request changed before grant"
    )
    busState := Mux(load, response, busIdle)
  }
  when(busState === response && native.io.mem_ready_o)(busState             := busIdle)

  blink.cmdReq.ready                                                                     := state === idle
  blink.cmdResp.valid                                                                    := state === complete
  blink.cmdResp.bits.rob_id                                                              := issue.rob_id
  blink.status.idle                                                                      := state === idle
  blink.status.running                                                                   := state =/= idle
  when(blink.cmdReq.fire) {
    val cmd = blink.cmdReq.bits
    val m   = cmd.rs2(11, 0)
    assert(
      cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 1.U,
      "REDMULE_F16 invalid bank groups"
    )
    assert(
      cmd.rs2(63, 43) === 0.U && m =/= 0.U && cmd.iter =/= 0.U && cmd.iter(2, 0) === 0.U,
      "REDMULE_F16 invalid dimensions or reserved bits"
    )
    assert(
      cmd.op1_bank =/= cmd.op2_bank && cmd.op1_bank =/= cmd.wr_bank && cmd.op2_bank =/= cmd.wr_bank,
      "REDMULE_F16 banks must differ"
    )
    for (
      (bank, footprint) <- Seq(
                             (cmd.op1_bank, cmd.rs2(21, 12) +& m * (cmd.iter >> 3)),
                             (cmd.op2_bank, cmd.rs2(31, 22) +& cmd.iter),
                             (cmd.wr_bank, cmd.rs2(41, 32) +& m)
                           )
    ) {
      val shared   = bank >= fp.shared_bank_id_base.U
      val capacity = Mux(
        shared,
        mp.sharedBankEntries.U,
        mp.bankEntries.U
      )
      assert(footprint <= capacity, "REDMULE_F16 bank window exceeds capacity")
    }
    issue := blink.cmdReq.bits
    finished    := false.B
    controlStep := 0.U
    state       := channels
  }
  when(state === channels)(state                                                         := configure)
  when(state === configure && native.io.ctrl_gnt_o)(state                                := controlReply)
  when(state === controlReply && native.io.ctrl_valid_o) {
    when(controlStep === 0.U) {
      assert(native.io.ctrl_data_o =/= "hffffffff".U, "RedMulE has no free job context")
    }
    when(controlStep === 8.U)(state := running).otherwise {
      controlStep := controlStep + 1.U
      state       := configure
    }
  }
  when(state === running && native.io.evt_o)(finished                                    := true.B)
  when(state === running && finished && !native.io.busy_o && busState === busIdle)(state := complete)
  when(blink.cmdResp.fire)(state                                                         := idle)
}

object RedmuleBall extends BallFactory {

  def instantiate(
    fp: FrontendParam,
    mp: MemDomainParam,
    bp: BallDomainParam,
    id: Int
  ): BallPorts =
    BallPorts(Instantiate(new RedmuleBall(fp, mp, bp, id)).blink, None)

}
