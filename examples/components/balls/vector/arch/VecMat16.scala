package examples.balls.vector

import chisel3.experimental.hierarchy.public

import chisel3.experimental.hierarchy.instantiable

import chisel3.experimental.hierarchy.{Instance, Instantiate}

import framework.acc.balldomain.configs.BallDomainParam
import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import examples.balls.vector.configs.VectorBallParam

import chisel3._
import framework.acc.balldomain.ball.blink.{BallCommand, BallResponse}
import chisel3.util._
import framework.acc.balldomain.ball.blink.BallStatus
import framework.acc.memdomain.{BankRead, BankWrite}

@instantiable
class VecMat16(
  frontendParam: FrontendParam,
  memParam:      MemDomainParam,
  routingParam:  BallDomainParam,
  val inBW:      Int,
  val outBW:     Int,
  val inW:       Int,
  val accW:      Int,
  lane:          Int,
  val ballParam: VectorBallParam)
    extends Module {
  val commandDataBits = frontendParam.commandDataBits

  require(lane == 16, "VecMat16 requires lane=16")
  require(inW == 8, "VecMat16 requires int8 inputs")
  require(accW == 32, "VecMat16 requires int32 accumulators")
  require(inBW >= 2, "VecMat16 requires two read ports")
  require(outBW >= 4, "VecMat16 requires four write ports")

  @public
  val io = IO(new Bundle {

    val cmdReq = Flipped(Decoupled(new BallCommand(frontendParam, memParam)))

    val cmdResp = Decoupled(new BallResponse(frontendParam))

    val bankRead = Vec(
      inBW,
      Flipped(new BankRead(
        frontendParam,
        memParam
      ))
    )

    val bankWrite = Vec(
      outBW,
      Flipped(new BankWrite(
        frontendParam,
        memParam
      ))
    )

    val status = new BallStatus
  })

  val sIdle :: sLoadAcc :: sReadReq :: sReadResp :: sWrite :: sDone :: Nil =
    Enum(6)
  val state                                                                = RegInit(sIdle)

  val robIdReg  = RegInit(0.U(log2Up(frontendParam.rob_entries).W))
  val op1Bank   = RegInit(0.U(frontendParam.bank_id_len.W))
  val op2Bank   = RegInit(0.U(frontendParam.bank_id_len.W))
  val wrBank    = RegInit(0.U(frontendParam.bank_id_len.W))
  val iterReg   = RegInit(0.U(frontendParam.iter_len.W))
  val readRow   = RegInit(0.U(frontendParam.iter_len.W))
  val aReqDone  = RegInit(false.B)
  val bReqDone  = RegInit(false.B)
  val aRespDone = RegInit(false.B)
  val bRespDone = RegInit(false.B)
  val aWord     = RegInit(0.U(memParam.bankWidth.W))
  val bWord     = RegInit(0.U(memParam.bankWidth.W))
  val acc       = Reg(Vec(lane, Vec(lane, UInt(accW.W))))

  val accLoad = Instantiate(new VecMat16AccLoad(frontendParam, memParam, routingParam, accW, lane, ballParam))

  val store = Instantiate(new VecMat16Store(frontendParam, memParam, routingParam, accW, lane, ballParam))

  accLoad.io.start  := false.B
  accLoad.io.wrBank := wrBank
  accLoad.io.robId  := robIdReg
  store.io.start    := false.B
  store.io.wrBank   := wrBank
  store.io.robId    := robIdReg
  store.io.acc      := acc
  for (p <- 0 until 2) {
    accLoad.io.bankRead(p).io.req.ready  := io.bankRead(p).io.req.ready
    accLoad.io.bankRead(p).io.resp.valid := io.bankRead(p).io.resp.valid
    accLoad.io.bankRead(p).io.resp.bits  := io.bankRead(p).io.resp.bits
  }
  for (g <- 0 until 4) {
    store.io.bankWrite(g).io.req.ready  := io.bankWrite(g).io.req.ready
    store.io.bankWrite(g).io.resp.valid := io.bankWrite(g).io.resp.valid
    store.io.bankWrite(g).io.resp.bits  := io.bankWrite(g).io.resp.bits
  }

  io.cmdReq.ready        := state === sIdle
  io.cmdResp.valid       := state === sDone
  io.cmdResp.bits.rob_id := robIdReg
  io.status.idle         := state === sIdle
  io.status.running      := state =/= sIdle && state =/= sDone
  VecMat16Tie.offRead(io.bankRead, robIdReg)
  VecMat16Tie.offWrite(io.bankWrite, robIdReg, wrBank, memParam)

  switch(state) {
    is(sIdle) {
      when(io.cmdReq.fire) {
        val cmd = io.cmdReq.bits
        robIdReg         := io.cmdReq.bits.rob_id
        op1Bank          := cmd.op1_bank
        op2Bank          := cmd.op2_bank
        wrBank           := cmd.wr_bank
        iterReg          := cmd.iter
        readRow          := 0.U
        aReqDone         := false.B
        bReqDone         := false.B
        aRespDone        := false.B
        bRespDone        := false.B
        assert(cmd.iter > 0.U, "VecMat16 iter must be > 0")
        assert(
          cmd.op1_col === 1.U && cmd.op2_col === 1.U && cmd.wr_col === 4.U,
          "VecMat16 unsupported bank layout"
        )
        accLoad.io.start := true.B
        state            := sLoadAcc
      }
    }

    is(sLoadAcc) {
      for (p <- 0 until 2) {
        VecMat16Tie.takeRead(io.bankRead(p), accLoad.io.bankRead(p))
      }
      when(accLoad.io.done) {
        acc      := accLoad.io.acc
        aReqDone := false.B
        bReqDone := false.B
        state    := sReadReq
      }
    }

    is(sReadReq) {
      io.bankRead(0).bank_id                    := op1Bank
      io.bankRead(0).io.req.valid               := !aReqDone
      io.bankRead(0).io.req.bits.addr           := readRow
      io.bankRead(1).bank_id                    := op2Bank
      io.bankRead(1).io.req.valid               := !bReqDone
      io.bankRead(1).io.req.bits.addr           := readRow
      when(io.bankRead(0).io.req.fire)(aReqDone := true.B)
      when(io.bankRead(1).io.req.fire)(bReqDone := true.B)
      when(
        (aReqDone || io
          .bankRead(0)
          .io
          .req
          .fire) && (bReqDone || io.bankRead(1).io.req.fire)
      ) {
        aRespDone := false.B
        bRespDone := false.B
        state     := sReadResp
      }
    }

    is(sReadResp) {
      io.bankRead(0).bank_id       := op1Bank
      io.bankRead(1).bank_id       := op2Bank
      io.bankRead(0).io.resp.ready := !aRespDone
      io.bankRead(1).io.resp.ready := !bRespDone
      val aFire = io.bankRead(0).io.resp.fire
      val bFire = io.bankRead(1).io.resp.fire
      val nextA = Mux(aFire, io.bankRead(0).io.resp.bits.data, aWord)
      val nextB = Mux(bFire, io.bankRead(1).io.resp.bits.data, bWord)
      val haveA = aRespDone || aFire
      val haveB = bRespDone || bFire
      when(aFire)(aWord     := io.bankRead(0).io.resp.bits.data)
      when(bFire)(bWord     := io.bankRead(1).io.resp.bits.data)
      when(haveA)(aRespDone := true.B)
      when(haveB)(bRespDone := true.B)
      when(haveA && haveB) {
        val aElems = nextA.asTypeOf(Vec(lane, UInt(inW.W)))
        val bElems = nextB.asTypeOf(Vec(lane, UInt(inW.W)))
        for {
          i <- 0 until lane
          j <- 0 until lane
        } {
          acc(i)(j) := (acc(i)(j).asSInt + aElems(i).asSInt * bElems(
            j
          ).asSInt).asUInt
        }
        when(readRow === iterReg - 1.U) {
          store.io.start := true.B
          state          := sWrite
        }.otherwise {
          readRow  := readRow + 1.U
          aReqDone := false.B
          bReqDone := false.B
          state    := sReadReq
        }
      }
    }

    is(sWrite) {
      for (g <- 0 until 4) {
        VecMat16Tie.takeWrite(io.bankWrite(g), store.io.bankWrite(g))
      }
      when(store.io.done)(state := sDone)
    }

    is(sDone) {
      when(io.cmdResp.fire)(state := sIdle)
    }
  }
}
