package examples.balls.vector

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import examples.balls.vector.configs.VectorBallParam
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.top.GlobalConfig

@instantiable
class VecUnit(val b: GlobalConfig) extends Module {
  val cfg        = VectorBallParam()
  val lane       = cfg.lane
  val inputWidth = cfg.inputWidth
  val accWidth   = cfg.outputWidth

  require(lane == 16, "VecUnit mul_warp16 requires lane=16")
  require(inputWidth == 8, "VecUnit mul_warp16 requires int8 inputs")
  require(accWidth == 32, "VecUnit mul_warp16 requires int32 accumulators")

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "VecBall")
    .getOrElse(throw new IllegalArgumentException("VecBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  require(inBW >= 2, "VecUnit mul_warp16 requires two read ports")
  require(outBW >= 4, "VecUnit mul_warp16 requires four write ports")

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val sIdle :: sReadReq :: sReadResp :: sWriteReq :: sWriteResp :: sDone :: Nil = Enum(6)
  val state                                                                     = RegInit(sIdle)

  val robIdReg    = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val isSubReg    = RegInit(false.B)
  val subRobIdReg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  val op1Bank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op2Bank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wrBank  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iterReg = RegInit(0.U(b.frontend.iter_len.W))

  val readRow  = RegInit(0.U(b.frontend.iter_len.W))
  val writeRow = RegInit(0.U(log2Ceil(lane + 1).W))

  val aReqDone    = RegInit(false.B)
  val bReqDone    = RegInit(false.B)
  val aRespDone   = RegInit(false.B)
  val bRespDone   = RegInit(false.B)
  val aWord       = RegInit(0.U(b.memDomain.bankWidth.W))
  val bWord       = RegInit(0.U(b.memDomain.bankWidth.W))
  val writeIssued = RegInit(VecInit(Seq.fill(4)(false.B)))
  val writeAcked  = RegInit(VecInit(Seq.fill(4)(false.B)))

  val acc = Reg(Vec(lane, Vec(lane, UInt(accWidth.W))))

  io.cmdReq.ready            := state === sIdle
  io.cmdResp.valid           := state === sDone
  io.cmdResp.bits.rob_id     := robIdReg
  io.cmdResp.bits.is_sub     := isSubReg
  io.cmdResp.bits.sub_rob_id := subRobIdReg

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id           := robIdReg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := 0.U
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id           := robIdReg
    io.bankWrite(i).ball_id          := 0.U
    io.bankWrite(i).bank_id          := wrBank
    io.bankWrite(i).group_id         := i.U
    io.bankWrite(i).io.req.valid     := false.B
    io.bankWrite(i).io.req.bits.addr := 0.U
    io.bankWrite(i).io.req.bits.data := 0.U
    io.bankWrite(i).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    io.bankWrite(i).io.resp.ready    := false.B
  }

  def rowVec(word: UInt): Vec[UInt] =
    word.asTypeOf(Vec(lane, UInt(inputWidth.W)))

  def writeData(group: Int): UInt = {
    val row   = writeRow(log2Ceil(lane) - 1, 0)
    val elems = (0 until 4).map { laneIdx =>
      acc(row)(group * 4 + laneIdx)
    }
    Cat(elems.reverse)
  }

  switch(state) {
    is(sIdle) {
      when(io.cmdReq.fire) {
        robIdReg    := io.cmdReq.bits.rob_id
        isSubReg    := io.cmdReq.bits.is_sub
        subRobIdReg := io.cmdReq.bits.sub_rob_id
        op1Bank     := io.cmdReq.bits.cmd.op1_bank
        op2Bank     := io.cmdReq.bits.cmd.op2_bank
        wrBank      := io.cmdReq.bits.cmd.wr_bank
        iterReg     := io.cmdReq.bits.cmd.iter
        readRow     := 0.U
        writeRow    := 0.U
        aReqDone    := false.B
        bReqDone    := false.B
        aRespDone   := false.B
        bRespDone   := false.B
        writeIssued := VecInit(Seq.fill(4)(false.B))
        writeAcked  := VecInit(Seq.fill(4)(false.B))
        for (i <- 0 until lane) {
          for (j <- 0 until lane) {
            acc(i)(j) := 0.U
          }
        }
        assert(io.cmdReq.bits.cmd.iter > 0.U, "VecUnit mul_warp16 iter must be > 0")
        assert(
          io.cmdReq.bits.cmd.op1_col === 1.U && io.cmdReq.bits.cmd.op2_col === 1.U &&
            io.cmdReq.bits.cmd.wr_col === 4.U,
          "VecUnit mul_warp16 unsupported bank layout"
        )
        state := sReadReq
      }
    }

    is(sReadReq) {
      io.bankRead(0).bank_id          := op1Bank
      io.bankRead(0).io.req.valid     := !aReqDone
      io.bankRead(0).io.req.bits.addr := readRow
      io.bankRead(1).bank_id          := op2Bank
      io.bankRead(1).io.req.valid     := !bReqDone
      io.bankRead(1).io.req.bits.addr := readRow

      when(io.bankRead(0).io.req.fire) {
        aReqDone := true.B
      }
      when(io.bankRead(1).io.req.fire) {
        bReqDone := true.B
      }
      when((aReqDone || io.bankRead(0).io.req.fire) && (bReqDone || io.bankRead(1).io.req.fire)) {
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

      val aFire  = io.bankRead(0).io.resp.fire
      val bFire  = io.bankRead(1).io.resp.fire
      val nextA  = Mux(aFire, io.bankRead(0).io.resp.bits.data, aWord)
      val nextB  = Mux(bFire, io.bankRead(1).io.resp.bits.data, bWord)
      val haveA  = aRespDone || aFire
      val haveB  = bRespDone || bFire
      val aElems = rowVec(nextA)
      val bElems = rowVec(nextB)

      when(aFire) {
        aWord := io.bankRead(0).io.resp.bits.data
      }
      when(bFire) {
        bWord := io.bankRead(1).io.resp.bits.data
      }
      when(haveA) {
        aRespDone := true.B
      }
      when(haveB) {
        bRespDone := true.B
      }

      when(haveA && haveB) {
        for (i <- 0 until lane) {
          for (j <- 0 until lane) {
            val prod = aElems(i).asSInt * bElems(j).asSInt
            acc(i)(j) := (acc(i)(j).asSInt + prod).asUInt
          }
        }

        when(readRow === iterReg - 1.U) {
          writeRow    := 0.U
          writeIssued := VecInit(Seq.fill(4)(false.B))
          writeAcked  := VecInit(Seq.fill(4)(false.B))
          state       := sWriteReq
        }.otherwise {
          readRow  := readRow + 1.U
          aReqDone := false.B
          bReqDone := false.B
          state    := sReadReq
        }
      }
    }

    is(sWriteReq) {
      for (group <- 0 until 4) {
        io.bankWrite(group).bank_id          := wrBank
        io.bankWrite(group).group_id         := group.U
        io.bankWrite(group).io.req.valid     := !writeIssued(group)
        io.bankWrite(group).io.req.bits.addr := writeRow
        io.bankWrite(group).io.req.bits.data := writeData(group)
        io.bankWrite(group).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))

        when(io.bankWrite(group).io.req.fire) {
          writeIssued(group) := true.B
        }
      }

      val allIssued = (0 until 4)
        .map(group => writeIssued(group) || io.bankWrite(group).io.req.fire)
        .reduce(_ && _)
      when(allIssued) {
        writeAcked := VecInit(Seq.fill(4)(false.B))
        state      := sWriteResp
      }
    }

    is(sWriteResp) {
      for (group <- 0 until 4) {
        io.bankWrite(group).bank_id       := wrBank
        io.bankWrite(group).group_id      := group.U
        io.bankWrite(group).io.resp.ready := !writeAcked(group)

        when(io.bankWrite(group).io.resp.fire) {
          writeAcked(group) := true.B
        }
      }

      val allAcked = (0 until 4)
        .map(group => writeAcked(group) || io.bankWrite(group).io.resp.fire)
        .reduce(_ && _)
      when(allAcked) {
        when(writeRow === (lane - 1).U) {
          state := sDone
        }.otherwise {
          writeRow    := writeRow + 1.U
          writeIssued := VecInit(Seq.fill(4)(false.B))
          writeAcked  := VecInit(Seq.fill(4)(false.B))
          state       := sWriteReq
        }
      }
    }

    is(sDone) {
      when(io.cmdResp.fire) {
        state := sIdle
      }
    }
  }

  io.status.idle    := state === sIdle
  io.status.running := state =/= sIdle && state =/= sDone
}
