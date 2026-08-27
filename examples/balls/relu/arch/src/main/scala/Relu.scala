package examples.balls.relu

import chisel3._
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class Relu(val b: GlobalConfig) extends Module {
  private val addressWidth = log2Up(b.memDomain.bankEntries)
  private val countWidth   = log2Up(b.memDomain.bankEntries + 1)
  private val bankWidth    = log2Up(b.memDomain.bankNum)

  private val mapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "ReluBall")
    .getOrElse(throw new IllegalArgumentException("ReluBall not found in config"))

  private val funct = b.ballDomain.ballISA
    .find(_.mnemonic == "RELU")
    .map(_.funct7)
    .getOrElse(throw new IllegalArgumentException("RELU not found in ballISA"))

  require(mapping.inBW == 1, "ReluBall requires one SRAM read port")
  require(mapping.outBW == 1, "ReluBall requires one SRAM write port")
  require(b.memDomain.bankWidth == 128, "ReluBall requires 128-bit SRAM rows")
  require(b.memDomain.bankMaskLen == 16, "ReluBall requires sixteen byte enables")

  @public
  val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val channelReady = Input(Bool())
    val bankRead     = Vec(1, Flipped(new BankRead(b)))
    val bankWrite    = Vec(1, Flipped(new BankWrite(b)))
    val status       = new BallStatus
  })

  val Seq(idle, waitForChannels, readRequest, readResponse, writeRequest, writeResponse, complete) = Enum(7)
  val state                                                                                        = RegInit(idle)

  val robId    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSub    = RegInit(false.B)
  val subRobId = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  val bank     = RegInit(0.U(bankWidth.W))
  val group    = RegInit(0.U(bankWidth.W))
  val iter     = RegInit(0.U(countWidth.W))
  val stride   = RegInit(0.U(countWidth.W))
  val segment  = RegInit(0.U(addressWidth.W))
  val line     = RegInit(0.U(addressWidth.W))
  val word     = Reg(UInt(128.W))

  val relu = Wire(Vec(4, UInt(32.W)))
  for (lane <- 0 until 4) {
    val value = word(32 * lane + 31, 32 * lane).asSInt
    relu(lane) := Mux(value < 0.S, 0.U, value.asUInt)
  }
  val address = segment + line

  io.cmdReq.ready            := state === idle
  io.cmdResp.valid           := state === complete
  io.cmdResp.bits.rob_id     := robId
  io.cmdResp.bits.is_sub     := isSub
  io.cmdResp.bits.sub_rob_id := subRobId

  io.bankRead(0).rob_id           := robId
  io.bankRead(0).ball_id          := 0.U
  io.bankRead(0).bank_id          := bank
  io.bankRead(0).group_id         := group
  io.bankRead(0).io.req.valid     := false.B
  io.bankRead(0).io.req.bits.addr := address
  io.bankRead(0).io.resp.ready    := false.B

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).ball_id          := 0.U
  io.bankWrite(0).bank_id          := bank
  io.bankWrite(0).group_id         := group
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := address
  io.bankWrite(0).io.req.bits.data := Cat(relu.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val command = io.cmdReq.bits.cmd
        assert(command.funct7 === funct.U(7.W), "ReluBall funct7 must be RELU")
        assert(command.wr_bank === 0.U, "ReluBall bank2 must be zero")
        assert(command.op1_col =/= 0.U && command.op2_bank < command.op1_col, "ReluBall group is not allocated in bank")
        assert(command.iter =/= 0.U && command.iter(3, 0) === 0.U, "ReluBall iter must be a positive multiple of 16")
        assert(
          command.special =/= 0.U && command.special <= b.memDomain.bankEntries.U,
          "ReluBall stride must fit in one physical bank"
        )
        assert(
          b.memDomain.bankEntries.U % command.special === 0.U,
          "ReluBall stride must divide the physical bank depth"
        )
        assert(command.iter <= command.special, "ReluBall iter must not exceed stride")

        robId    := io.cmdReq.bits.rob_id
        isSub    := io.cmdReq.bits.is_sub
        subRobId := io.cmdReq.bits.sub_rob_id
        bank     := command.op1_bank
        group    := command.op2_bank
        iter     := command.iter
        stride   := command.special
        segment  := 0.U
        line     := 0.U
        state    := waitForChannels
      }
    }

    is(waitForChannels) {
      when(io.channelReady)(state := readRequest)
    }

    is(readRequest) {
      io.bankRead(0).io.req.valid            := true.B
      when(io.bankRead(0).io.req.fire)(state := readResponse)
    }

    is(readResponse) {
      io.bankRead(0).io.resp.ready := true.B
      when(io.bankRead(0).io.resp.fire) {
        word  := io.bankRead(0).io.resp.bits.data
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
        when(line +& 1.U === iter) {
          when(segment +& stride === b.memDomain.bankEntries.U) {
            state := complete
          }.otherwise {
            segment := segment + stride
            line    := 0.U
            state   := readRequest
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
