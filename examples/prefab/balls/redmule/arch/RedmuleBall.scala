package examples.balls.redmule

import framework.generic.contract.blink.BlinkParam
import framework.generic.contract.configs.BallRouteParam
import framework.ball.sub_rob.ExpansionParam

import framework.ball.bbus.{BallFactory, BallPorts}
import buckyball.balls.redmule.{Params => ParametersProto}
import chisel3.experimental.hierarchy.{Instance, Instantiate}

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.ball.blink.{BallStatus, BlinkIO, HasBlink}
import framework.generic.contract.blink.BallCommand

@instantiable
class RedmuleBall(
  link:  BlinkParam,
  route: BallRouteParam)
    extends Module
    with HasBlink {
  val ball = route

  ball.parameters.unpack(classOf[ParametersProto])

  require(ball.inBW == 2 && ball.outBW == 1)
  require(
    link.bankWidth == 128 && link.bankMaskLen == 16
  )
  require(link.bankEntries <= 65535)

  @public
  val blink = IO(new BlinkIO(link, route))

  val Seq(idle, channels, configure, controlReply, running, complete) = Enum(6)
  val state                                                           = RegInit(idle)

  val issue = Reg(new BallCommand(link))

  val controlStep = RegInit(0.U(4.W))
  val finished    = RegInit(false.B)
  val rows        = issue.rs2(11, 0)
  val aBase       = issue.rs2(21, 12)
  val bBase       = issue.rs2(31, 22)
  val cBase       = issue.rs2(41, 32)
  val first       = issue.rs2(42)
  // One native job per row tile of Width = 8 C rows: every job starts from
  // the same native state, so the bus timing of a tile does not depend on
  // its neighbours, and the native memory scheduler never sees more than
  // one tile per job.
  val tile        = RegInit(0.U(9.W))
  val left        = rows - (tile << 3)
  val tileRows    = Mux(left > 8.U, 8.U, left)
  val tileA       = aBase +& tile * issue.iter(15, 0)
  val tileC       = cBase +& (tile << 3)
  val native      = Instantiate(new RedmuleNative)
  native.io.clk_i       := clock
  native.io.rst_ni      := !reset.asBool
  native.io.ctrl_req_i  := state === configure
  // Step 0 soft-clears the engine state (SOFT_CLEAR bit 1 = 0, bit 0 = 1 keeps
  // the registers): every job, also every tile, starts with the same native
  // state, the round-robin counter of its memory mux included.
  native.io.ctrl_wen_i  := controlStep === 1.U
  native.io.ctrl_addr_i := VecInit(Seq(20, 4, 44, 48, 52, 32, 36, 40, 60, 0).map(_.U(32.W)))(controlStep)
  native.io.ctrl_data_i := VecInit(Seq(
    1.U(32.W),
    0.U(32.W),
    16777216.U | (tileA << 4),
    33554432.U | (bBase << 4),
    50331648.U | (tileC << 4),
    (8.U(32.W) << 16) | tileRows,
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
  val startRow                                                                                 = Mux(region === 1.U, tileA, Mux(region === 2.U, bBase, tileC))
  val span                                                                                     = Mux(region === 1.U, tileRows * (issue.iter(15, 0) >> 3), Mux(region === 2.U, issue.iter(15, 0), tileRows))
  val inWindow                                                                                 = bankRow >= startRow && bankRow < (startRow +& span)
  val readPort                                                                                 = Mux(region === 2.U, 1.U, 0.U)

  // A continuation reads the previous C from the bank the destination held before this write.
  val readBank = Mux(
    region === 1.U,
    issue.op1_bank,
    Mux(region === 2.U, issue.op2_bank, issue.wr_read_bank)
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

  blink.cmdReq.ready                                      := state === idle
  blink.cmdResp.valid                                     := state === complete
  blink.cmdResp.bits.rob_id                               := issue.rob_id
  blink.status.idle                                       := state === idle
  blink.status.running                                    := state =/= idle
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
      footprint <- Seq(
                     cmd.rs2(21, 12) +& m * (cmd.iter >> 3),
                     cmd.rs2(31, 22) +& cmd.iter,
                     cmd.rs2(41, 32) +& m
                   )
    ) {
      assert(footprint <= link.bankEntries.U, "REDMULE_F16 bank window exceeds capacity")
    }
    issue       := blink.cmdReq.bits
    finished    := false.B
    controlStep := 0.U
    tile        := 0.U
    state       := channels
  }
  when(state === channels)(state                          := configure)
  when(state === configure && native.io.ctrl_gnt_o)(state := controlReply)
  when(state === controlReply && native.io.ctrl_valid_o) {
    when(controlStep === 1.U) {
      assert(native.io.ctrl_data_o =/= "hffffffff".U, "RedMulE has no free job context")
    }
    when(controlStep === 9.U)(state := running).otherwise {
      controlStep := controlStep + 1.U
      state       := configure
    }
  }
  when(state === running && native.io.evt_o)(finished     := true.B)
  when(state === running && finished && !native.io.busy_o && busState === busIdle) {
    when(left > 8.U) {
      tile        := tile + 1.U
      finished    := false.B
      controlStep := 0.U
      state       := configure
    }.otherwise(state := complete)
  }
  when(blink.cmdResp.fire)(state                          := idle)
}

object RedmuleBall extends BallFactory {

  def instantiate(
    link:      BlinkParam,
    route:     BallRouteParam,
    expansion: Option[ExpansionParam]
  ): BallPorts =
    BallPorts(Instantiate(new RedmuleBall(link, route)).blink, None)

}
