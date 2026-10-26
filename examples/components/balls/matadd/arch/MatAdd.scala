package examples.balls.matadd

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
class MatAdd(
  val frontendParam: FrontendParam,
  val memParam:      MemDomainParam,
  val routingParam:  BallDomainParam,
  val instance:      BallRouteParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits
  val addressWidth    = log2Up(memParam.bankEntries)
  val countWidth      = log2Up(memParam.bankEntries + 1)
  val bankIdWidth     = frontendParam.bank_id_len

  val mapping = instance

  require(mapping.inBW == 2, "MatAddBall requires two SRAM read ports")
  require(mapping.outBW == 1, "MatAddBall requires one SRAM write port")
  require(memParam.bankWidth == 128, "MatAddBall requires 128-bit SRAM rows")
  require(
    memParam.bankMaskLen == 16,
    "MatAddBall requires sixteen byte enables"
  )

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      2,
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

  val robId        = RegInit(0.U(log2Up(frontendParam.rob_entries).W))
  val aBank        = RegInit(0.U(bankIdWidth.W))
  val bBank        = RegInit(0.U(bankIdWidth.W))
  val cBank        = RegInit(0.U(bankIdWidth.W))
  val groups       = RegInit(0.U(5.W))
  val group        = RegInit(0.U(5.W))
  val bases        = Reg(Vec(3, UInt(memParam.groupIdWidth.W)))
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

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := state === complete
  io.cmdResp.bits.rob_id := robId

  for (port <- 0 until 2) {
    io.bankRead(port).rob_id           := robId
    io.bankRead(port).bank_id          := (if (port == 0) aBank else bBank)
    io.bankRead(port).group_id         := bases(port) + group
    io.bankRead(port).io.req.valid     := false.B
    io.bankRead(port).io.req.bits.addr := line
    io.bankRead(port).io.resp.ready    := false.B
  }

  io.bankWrite(0).rob_id           := robId
  io.bankWrite(0).bank_id          := cBank
  io.bankWrite(0).group_id         := bases(2) + group
  io.bankWrite(0).io.req.valid     := false.B
  io.bankWrite(0).io.req.bits.addr := line
  io.bankWrite(0).io.req.bits.data := Cat(sum.reverse)
  io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(16)(true.B))
  io.bankWrite(0).io.resp.ready    := false.B

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val command = io.cmdReq.bits

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
        assert(
          command.op1_col <= memParam.bankNum.U,
          "MatAddBall bank groups must fit in physical banks"
        )
        assert(
          command.iter =/= 0.U && command.iter <= memParam.bankEntries.U,
          "MatAddBall iter must fit in one physical bank"
        )
        assert(command.rs2 === 0.U, "MatAddBall reserves rs2")

        robId        := io.cmdReq.bits.rob_id
        aBank        := command.op1_bank
        bBank        := command.op2_bank
        cBank        := command.wr_bank
        groups       := command.op1_col
        group        := 0.U
        bases        := VecInit(command.op1_group, command.op2_group, command.wr_group)
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
      state := readRequest
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
