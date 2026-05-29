package framework.balldomain.prototype.transpose

import chisel3._
import chisel3.util._
import chisel3.stage._
import chisel3.experimental.hierarchy.{instantiable, public}

import framework.balldomain.prototype.vector._
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.top.GlobalConfig
import framework.balldomain.prototype.transpose.configs.TransposeBallParam

@instantiable
class Transpose(val b: GlobalConfig) extends Module {
  val ballConfig = TransposeBallParam()
  val InputNum   = ballConfig.InputNum
  val inputWidth = ballConfig.inputWidth
  val bankWidth  = b.memDomain.bankWidth

  val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "TransposeBall")
    .getOrElse(throw new IllegalArgumentException("TransposeBall not found in config"))

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  // -------------------------------
  // ROB / IDs
  // -------------------------------
  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))
  when(io.cmdReq.fire) {
    rob_id_reg     := io.cmdReq.bits.rob_id
    is_sub_reg     := io.cmdReq.bits.is_sub
    sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
  }

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  val idle :: read :: write :: complete :: Nil = Enum(4)
  val state                                    = RegInit(idle)
  val outRow                                   = RegInit(VecInit(Seq.fill(InputNum)(0.U(inputWidth.W))))

  // Command fields
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg  = RegInit(0.U(32.W))

  // Counters
  val outCol  = RegInit(0.U(32.W))
  val srcRow  = RegInit(0.U(log2Ceil(InputNum + 1).W))
  val pending = RegInit(false.B)

  // -------------------------------
  // Default IO assignments
  // -------------------------------
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id         := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
    io.bankWrite(i).bank_id           := wbank_reg
    io.bankWrite(i).group_id          := 0.U
  }

  io.cmdReq.ready            := (state === idle)
  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  io.bankRead(0).io.resp.ready  := (state === read) && pending
  io.bankWrite(0).io.resp.ready := (state =/= idle)

  val srcByte   = srcRow * iter_reg + outCol
  val srcLane   = srcByte(log2Ceil(InputNum) - 1, 0)
  val srcAddr   = srcByte >> log2Ceil(InputNum)
  val rowIdx    = srcRow(log2Ceil(InputNum) - 1, 0)
  val readLanes = io.bankRead(0).io.resp.bits.data.asTypeOf(Vec(InputNum, UInt(inputWidth.W)))
  val packedRow = Cat(outRow.reverse)

  // -------------------------------
  // Main FSM
  // -------------------------------
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val iterVal = io.cmdReq.bits.cmd.iter

        rbank_reg        := io.cmdReq.bits.cmd.op1_bank
        wbank_reg        := io.cmdReq.bits.cmd.wr_bank
        iter_reg         := iterVal
        outCol           := 0.U
        srcRow           := 0.U
        pending          := false.B
        outRow.foreach(_ := 0.U)
        assert(io.cmdReq.bits.cmd.iter > 0.U, "Transpose iter must be > 0")
        assert(
          io.cmdReq.bits.cmd.op1_col === 1.U && io.cmdReq.bits.cmd.wr_col === 1.U,
          "Transpose unsupported bank layout"
        )
        state            := read
      }
    }

    is(read) {
      io.bankRead(0).io.req.valid     := (srcRow < InputNum.U) && !pending
      io.bankRead(0).io.req.bits.addr := srcAddr
      when(io.bankRead(0).io.req.fire) {
        pending := true.B
      }

      when(io.bankRead(0).io.resp.fire) {
        outRow(rowIdx) := readLanes(srcLane)
        pending        := false.B
        srcRow         := srcRow + 1.U
        when(srcRow === (InputNum - 1).U) {
          state := write
        }
      }
    }

    is(write) {
      io.bankWrite(0).io.req.valid     := true.B
      io.bankWrite(0).io.req.bits.addr := outCol
      io.bankWrite(0).io.req.bits.data := packedRow
      io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))

      when(io.bankWrite(0).io.req.fire) {
        srcRow           := 0.U
        outRow.foreach(_ := 0.U)
        when(outCol + 1.U === iter_reg) {
          state := complete
        }.otherwise {
          outCol := outCol + 1.U
          state  := read
        }
      }
    }

    is(complete) {
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := (state === idle)
  io.status.running := (state =/= idle)
}
