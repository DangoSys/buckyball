package framework.balldomain.prototype.relu

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.prototype.vector._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.relu.configs.ReluBallParam

@instantiable
class PipelinedRelu(val b: GlobalConfig) extends Module {
  val ballConfig = ReluBallParam()
  val InputNum   = ballConfig.InputNum
  val inputWidth = ballConfig.inputWidth
  val bankWidth  = b.memDomain.bankWidth

  // Get bandwidth from config
  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "ReluBall")
    .getOrElse(throw new IllegalArgumentException("ReluBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire) {
    rob_id_reg := io.cmdReq.bits.rob_id
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  val idle :: sRead :: sWrite :: complete :: Nil = Enum(4)
  val state                                      = RegInit(idle)

  val regArray = RegInit(
    VecInit(Seq.fill(InputNum)(
      VecInit(Seq.fill(InputNum)(0.U(inputWidth.W)))
    ))
  )

  val readCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val respCounter  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val writeCounter = RegInit(0.U(log2Ceil(InputNum + 1).W))

  val waddr_reg    = RegInit(0.U(10.W))
  val wbank_reg    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val raddr_reg    = RegInit(0.U(10.W))
  val rbank_reg    = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg     = RegInit(0.U(10.W))
  val cycle_reg    = RegInit(0.U(6.W))
  val iterCnt      = RegInit(0.U(32.W))
  val writeDataReg = Reg(UInt(bankWidth.W))
  val writeMaskReg = Reg(Vec(b.memDomain.bankMaskLen, UInt(1.W)))

  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).bank_id      := rbank_reg
    io.bankRead(i).acc_group_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).bank_id      := wbank_reg
    io.bankWrite(i).acc_group_id := 0.U
  }

  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        state        := sRead
        readCounter  := 0.U
        respCounter  := 0.U
        writeCounter := 0.U
        waddr_reg    := 0.U
        wbank_reg    := io.cmdReq.bits.cmd.wr_bank
        raddr_reg    := 0.U
        rbank_reg    := io.cmdReq.bits.cmd.op1_bank
        iter_reg     := io.cmdReq.bits.cmd.iter
        cycle_reg    := (io.cmdReq.bits.cmd.iter +& (InputNum.U - 1.U)) / InputNum.U - 1.U
      }
      when(cycle_reg =/= 0.U) {
        state        := sRead
        readCounter  := 0.U
        writeCounter := 0.U
        respCounter  := 0.U
        waddr_reg    := waddr_reg + InputNum.U
        raddr_reg    := raddr_reg + InputNum.U
        cycle_reg    := cycle_reg - 1.U
      }
    }

    is(sRead) {
      io.bankRead(0).io.resp.ready := true.B

      io.bankRead(0).io.req.valid     := (readCounter < InputNum.U)
      io.bankRead(0).io.req.bits.addr := raddr_reg + readCounter

      when(io.bankRead(0).io.req.fire) {
        readCounter := readCounter + 1.U
      }

      val dataWord = io.bankRead(0).io.resp.bits.data

      when(io.bankRead(0).io.resp.fire) {
        for (col <- 0 until InputNum) {
          val hi     = (col + 1) * inputWidth - 1
          val lo     = col * inputWidth
          val raw    = dataWord(hi, lo)
          val signed = raw.asSInt
          val relu   = Mux(signed < 0.S, 0.S(inputWidth.W), signed)
          regArray(respCounter)(col) := relu.asUInt
        }
        respCounter := respCounter + 1.U
      }

      when(respCounter === (InputNum - 1).U) {
        state        := sWrite
        writeDataReg := Cat((0 until InputNum).reverse.map(j => regArray(0)(j)))
        for (i <- 0 until b.memDomain.bankMaskLen) {
          writeMaskReg(i) := 1.U(1.W)
        }
      }
    }

    is(sWrite) {
      val hasMore = writeCounter < InputNum.U

      io.bankWrite(0).io.req.valid     := hasMore
      io.bankWrite(0).io.req.bits.addr := waddr_reg + writeCounter
      io.bankWrite(0).io.req.bits.data := writeDataReg
      io.bankWrite(0).io.req.bits.mask := writeMaskReg
      io.bankWrite(0).io.resp.ready    := true.B

      when(io.bankWrite(0).io.req.fire) {
        when(writeCounter === (InputNum - 1).U) {
          state := complete
        }.otherwise {
          val nextCnt = writeCounter + 1.U
          writeCounter := nextCnt
          writeDataReg := Cat((0 until InputNum).reverse.map(j => regArray(nextCnt)(j)))
        }
      }
    }

    is(complete) {
      io.bankWrite(0).io.resp.ready := true.B
      when(cycle_reg === 0.U) {
        io.cmdResp.valid       := true.B
        io.cmdResp.bits.rob_id := rob_id_reg
        when(io.cmdResp.fire) {
          iterCnt := iterCnt + 1.U
        }
      }
      state                         := idle
    }
  }

  io.status.idle    := (state === idle)
  io.status.running := (state === sRead) || (state === sWrite)

  when(reset.asBool) {
    for (i <- 0 until InputNum) {
      for (j <- 0 until InputNum) {
        regArray(i)(j) := 0.U
      }
    }
    writeDataReg := 0.U
    for (i <- 0 until b.memDomain.bankMaskLen) {
      writeMaskReg(i) := 0.U
    }
  }
}
