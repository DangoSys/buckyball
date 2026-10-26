package examples.balls.relu

import framework.acc.balldomain.configs.BallRouteParam

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.experimental.hierarchy.{instantiable, public}
import chisel3.util._
import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}

@instantiable
class Relu(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val addressWidth    = log2Up(memParam.bankEntries)
  val countWidth      = log2Up(memParam.bankEntries + 1)
  val bankIdWidth     = frontendParam.bank_id_len
  val groupWidth      = log2Up(memParam.bankNum)

  val mapping = instance

  require(mapping.inBW == 1, "ReluBall requires one SRAM read port")
  require(mapping.outBW == 1, "ReluBall requires one SRAM write port")
  require(memParam.bankWidth == 128, "ReluBall requires 128-bit SRAM rows")
  require(memParam.bankMaskLen == 16, "ReluBall requires sixteen byte enables")

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      1,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val bankWrite = Vec(
      1,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val status = new BallStatus
  })

  val Seq(idle, waitForChannels, readRequest, readResponse, writeRequest, writeResponse, complete) = Enum(7)
  val state                                                                                        = RegInit(idle)

  val robId   = RegInit(0.U(log2Up(frontendParam.rob_entries).W))
  val bank    = RegInit(0.U(bankIdWidth.W))
  val group   = RegInit(0.U(groupWidth.W))
  val iter    = RegInit(0.U(countWidth.W))
  val stride  = RegInit(0.U(countWidth.W))
  val segment = RegInit(0.U(addressWidth.W))
  val line    = RegInit(0.U(addressWidth.W))
  val word    = Reg(UInt(128.W))

  val relu = Wire(Vec(4, UInt(32.W)))
  for (lane <- 0 until 4) {
    val value = word(32 * lane + 31, 32 * lane).asSInt
    relu(lane) := Mux(value < 0.S, 0.U, value.asUInt)
  }
  val address = segment + line

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := state === complete
  io.cmdResp.bits.rob_id := robId

  io.bankRead(0).rob_id           := robId
  io.bankRead(0).bank_id          := bank
  io.bankRead(0).group_id         := group
  io.bankRead(0).io.req.valid     := false.B
  io.bankRead(0).io.req.bits.addr := address
  io.bankRead(0).io.resp.ready    := false.B

  io.bankWrite(0).rob_id           := robId
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
        val command = io.cmdReq.bits

        assert(
          command.op1_bank === command.wr_bank,
          "ReluBall source and destination banks must match"
        )
        assert(
          command.op1_col =/= 0.U && command.op2_bank < command.op1_col,
          "ReluBall group is not allocated in bank"
        )
        assert(command.iter =/= 0.U && command.iter(3, 0) === 0.U, "ReluBall iter must be a positive multiple of 16")
        assert(
          command.rs2 =/= 0.U && command.rs2 <= memParam.bankEntries.U,
          "ReluBall stride must fit in one physical bank"
        )
        assert(
          memParam.bankEntries.U % command.rs2 === 0.U,
          "ReluBall stride must divide the physical bank depth"
        )
        assert(command.iter <= command.rs2, "ReluBall iter must not exceed stride")

        robId   := io.cmdReq.bits.rob_id
        bank    := command.op1_bank
        // RELU names its group in the op2 bank field.
        group   := command.op1_group + command.op2_bank
        iter    := command.iter
        stride  := command.rs2
        segment := 0.U
        line    := 0.U
        state   := waitForChannels
      }
    }

    is(waitForChannels) {
      state := readRequest
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
          when(segment +& stride === memParam.bankEntries.U) {
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
