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

  // -------------------------------
  // State: idle -> fill -> drain -> (fill or idle)
  // For 16xN with stride = N/16:
  //   fill(16) -> drain(16) -> fill(16) -> drain(16) -> ... -> idle
  // Total: 32 * stride cycles
  // -------------------------------
  val idle :: fill :: drain :: Nil = Enum(3)
  val state                        = RegInit(idle)

  // -------------------------------
  // Single 16x16 register array
  // -------------------------------
  val regArray = Reg(Vec(InputNum, Vec(InputNum, UInt(inputWidth.W))))

  // Command fields
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg  = RegInit(0.U(32.W))
  val stride    = RegInit(1.U(32.W)) // iter / InputNum = addresses per row

  // Counters
  val fillIdx  = RegInit(0.U(log2Ceil(InputNum + 1).W)) // row being filled (0..15)
  val drainIdx = RegInit(0.U(log2Ceil(InputNum + 1).W)) // column being drained (0..16)
  val round    = RegInit(0.U(32.W))                     // which column-group (0..stride-1)

  // Read request tracking
  val readReqCnt  = RegInit(0.U(32.W))
  val readRespCnt = RegInit(0.U(32.W))

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

  io.bankRead(0).io.resp.ready  := (state =/= idle)
  io.bankWrite(0).io.resp.ready := (state =/= idle)

  // -------------------------------
  // Helpers
  // -------------------------------
  // Read address: strided to gather one column-group
  // For round r, row i: addr = i * stride + r
  def readAddr(row: UInt, r: UInt): UInt = row * stride + r

  // Pack one column of regArray into a data word
  def packColumn(col: UInt): UInt = {
    Cat((0 until InputNum).reverse.map { r =>
      regArray(r.U)(col)
    })
  }

  // -------------------------------
  // Main FSM
  // -------------------------------
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        val iterVal   = io.cmdReq.bits.cmd.iter
        val strideVal = iterVal >> log2Ceil(InputNum)

        rbank_reg   := io.cmdReq.bits.cmd.op1_bank
        wbank_reg   := io.cmdReq.bits.cmd.wr_bank
        iter_reg    := iterVal
        stride      := Mux(strideVal === 0.U, 1.U, strideVal)
        round       := 0.U
        fillIdx     := 0.U
        drainIdx    := 0.U
        readReqCnt  := 0.U
        readRespCnt := 0.U
        state       := fill
      }
    }

    // -------------------------------------------------------
    // FILL: read 16 rows into regArray for current round
    // -------------------------------------------------------
    is(fill) {
      // Send read requests
      io.bankRead(0).io.req.valid     := (fillIdx < InputNum.U)
      io.bankRead(0).io.req.bits.addr := readAddr(fillIdx, round)
      when(io.bankRead(0).io.req.fire) {
        readReqCnt := readReqCnt + 1.U
        fillIdx    := fillIdx + 1.U
      }

      // Handle responses: fill regArray row by row
      when(io.bankRead(0).io.resp.fire) {
        val dataWord = io.bankRead(0).io.resp.bits.data
        val respRow  = readRespCnt(log2Ceil(InputNum) - 1, 0)
        for (col <- 0 until InputNum) {
          val hi = (col + 1) * inputWidth - 1
          val lo = col * inputWidth
          regArray(respRow)(col) := dataWord(hi, lo)
        }
        readRespCnt := readRespCnt + 1.U

        // All 16 responses received → buffer full, go to drain
        when(readRespCnt(log2Ceil(InputNum) - 1, 0) === (InputNum - 1).U) {
          drainIdx := 0.U
          state    := drain
        }
      }
    }

    // -------------------------------------------------------
    // DRAIN: write out 16 transposed columns, then fill next
    //        round or signal completion
    // -------------------------------------------------------
    is(drain) {
      when(drainIdx < InputNum.U) {
        io.bankWrite(0).io.req.valid     := true.B
        io.bankWrite(0).io.req.bits.addr := round * InputNum.U + drainIdx
        io.bankWrite(0).io.req.bits.data := packColumn(drainIdx)
        io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))

        when(io.bankWrite(0).io.req.fire) {
          drainIdx := drainIdx + 1.U
        }
      }.otherwise {
        // All 16 columns written
        when(round === stride - 1.U) {
          // Last round → signal completion
          io.cmdResp.valid       := true.B
          io.cmdResp.bits.rob_id := rob_id_reg
          when(io.cmdResp.fire) {
            state := idle
          }
        }.otherwise {
          // More rounds → advance round and fill next
          round   := round + 1.U
          fillIdx := 0.U
          state   := fill
        }
      }
    }
  }

  io.status.idle    := (state === idle)
  io.status.running := (state =/= idle)
}
