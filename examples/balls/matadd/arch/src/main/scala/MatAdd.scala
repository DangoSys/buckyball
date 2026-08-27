package examples.balls.matadd

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class MatAdd(val b: GlobalConfig) extends Module {
  private val addressWidth = log2Up(b.memDomain.bankEntries)
  private val countWidth   = log2Up(b.memDomain.bankEntries + 1)
  private val bankIdWidth  = log2Up(b.memDomain.bankNum)

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "MatAddBall")
    .getOrElse(throw new IllegalArgumentException("MatAddBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "MATADD")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("MATADD not found in ballISA"))

  require(mapping.inBW == 2, "MatAddBall requires two SRAM read ports")
  require(mapping.outBW == 1, "MatAddBall requires one SRAM write port")
  require(b.memDomain.bankWidth == 128, "MatAddBall requires 128-bit SRAM rows")
  require(b.memDomain.bankMaskLen == 16, "MatAddBall requires sixteen byte enables")

  @public
  val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val channelReady = Input(Bool())
    val bankRead     = Vec(2, Flipped(new BankRead(b)))
    val bankWrite    = Vec(1, Flipped(new BankWrite(b)))
    val status       = new BallStatus
  })

  val Seq(idle, waitForChannels, readRequest, readResponse, writeRequest, writeResponse, complete) = Enum(7)
  val state                                                                                        = RegInit(idle)

  val robId        = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSub        = RegInit(false.B)
  val subRobId     = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val aBank        = RegInit(0.U(bankIdWidth.W))
  val bBank        = RegInit(0.U(bankIdWidth.W))
  val cBank        = RegInit(0.U(bankIdWidth.W))
  val groups       = RegInit(0.U(5.W))
  val group        = RegInit(0.U(5.W))
  val iter         = RegInit(0.U(countWidth.W))
  val line         = RegInit(0.U(addressWidth.W))
  val aWord        = Reg(UInt(128.W))
  val bWord        = Reg(UInt(128.W))
  val aRequestSent = RegInit(false.B)
  val bRequestSent = RegInit(false.B)
  val aReceived    = RegInit(false.B)
  val bReceived    = RegInit(false.B)

  val sum = Wire(Vec(4, UInt(32.W)))
  for (lane <- 0 until 4) {
    sum(lane) := (aWord(32 * lane + 31, 32 * lane) +&
      bWord(32 * lane + 31, 32 * lane))(31, 0)
  }

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId

  for (port <- 0 until 2) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).ball_id          := 0.U
    io.bankRead(port).bank_id          := (if (port == 0) aBank else bBank)
    io.bankRead(port).group_id         := group
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := line
    io.bankRead(port).io.resp.ready    := false.B
  }

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := cBank
  io.bankWrite(0).group_id         := group
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := line
  io.bankWrite(0).io.req.bits.data := Cat(sum.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val command = io.cmdReq.bits.cmd
        assert(command.funct7 === funct.U(7.W), "MatAddBall funct7 must be MATADD")
        assert(command.op1_en && command.op2_en && command.wr_spad_en, "MatAddBall requires two inputs and one output")
        assert(
          command.op1_bank =/= command.op2_bank && command.op1_bank =/= command.wr_bank &&
            command.op2_bank =/= command.wr_bank,
          "MatAddBall banks must be distinct"
        )
        assert(
          command.op1_col =/= 0.U && command.op1_col === command.op2_col &&
            command.op1_col === command.wr_col,
          "MatAddBall bank groups must match"
        )
        assert(command.op1_col <= b.memDomain.bankNum.U, "MatAddBall bank groups must fit in physical banks")
        assert(
          command.iter =/= 0.U && command.iter <= b.memDomain.bankEntries.U,
          "MatAddBall iter must fit in one physical bank"
        )
        assert(command.special === 0.U, "MatAddBall reserves rs2")

        robId        := io.cmdReq.bits.rob_id
        isSub        := io.cmdReq.bits.is_sub
        subRobId     := io.cmdReq.bits.sub_rob_id
        aBank        := command.op1_bank
        bBank        := command.op2_bank
        cBank        := command.wr_bank
        groups       := command.op1_col
        group        := 0.U
        iter         := command.iter
        line         := 0.U
        aRequestSent := false.B
        bRequestSent := false.B
        aReceived    := false.B
        bReceived    := false.B
        state        := waitForChannels
      }
    }

    is(waitForChannels) {
      when(io.channelReady)(state := readRequest)
    }

    is(readRequest) {
      io.bankRead(0).io.req.valid                   := !aRequestSent
      io.bankRead(1).io.req.valid                   := !bRequestSent
      when(io.bankRead(0).io.req.fire)(aRequestSent := true.B)
      when(io.bankRead(1).io.req.fire)(bRequestSent := true.B)
      when((aRequestSent || io.bankRead(0).io.req.fire) &&
        (bRequestSent || io.bankRead(1).io.req.fire)) {
        state := readResponse
      }
    }

    is(readResponse) {
      io.bankRead(0).io.resp.ready := !aReceived
      io.bankRead(1).io.resp.ready := !bReceived
      when(io.bankRead(0).io.resp.fire) {
        aWord     := io.bankRead(0).io.resp.bits.data
        aReceived := true.B
      }
      when(io.bankRead(1).io.resp.fire) {
        bWord     := io.bankRead(1).io.resp.bits.data
        bReceived := true.B
      }
      when((aReceived || io.bankRead(0).io.resp.fire) &&
        (bReceived || io.bankRead(1).io.resp.fire)) {
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
        aRequestSent := false.B
        bRequestSent := false.B
        aReceived    := false.B
        bReceived    := false.B
        when(line +& 1.U === iter) {
          line := 0.U
          when(group +& 1.U === groups) {
            state := complete
          }.otherwise {
            group := group + 1.U
            state := readRequest
          }
        }.otherwise {
          line  := line + 1.U
          state := readRequest
        }
      }
    }

    is(complete) {
      when(io.cmdResp.fire)(state := idle)
    }
  }

  io.status.idle    := state === idle
  io.status.running := state =/= idle
}
