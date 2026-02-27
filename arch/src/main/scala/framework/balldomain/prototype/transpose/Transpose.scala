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
  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  when(io.cmdReq.fire)(rob_id_reg := io.cmdReq.bits.rob_id)

  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id  := rob_id_reg
    io.bankRead(i).ball_id := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id  := rob_id_reg
    io.bankWrite(i).ball_id := 0.U
  }

  // -------------------------------
  // State
  // -------------------------------
  val idle :: compute :: Nil = Enum(2)
  val state                  = RegInit(idle)

  // -------------------------------
  // Ping-pong row buffers:
  // 2 blocks, each stores InputNum rows, each row has InputNum elems
  // -------------------------------
  val regArray = Reg(Vec(2 * InputNum, Vec(InputNum, UInt(inputWidth.W))))

  // which block are we filling / draining
  val fillSel     = RegInit(0.U(1.W))                      // 0 or 1
  val drainSel    = RegInit(0.U(1.W))                      // 0 or 1
  val blockFull   = RegInit(VecInit(Seq.fill(2)(false.B)))
  val fillRowIdx  = RegInit(0.U(log2Ceil(InputNum + 1).W)) // 0..InputNum
  val drainColIdx = RegInit(0.U(log2Ceil(InputNum + 1).W)) // 0..InputNum
  val draining    = RegInit(false.B)

  // total progress counters (STRICTLY via fire)
  val readReqCnt  = RegInit(0.U(32.W))
  val readRespCnt = RegInit(0.U(32.W))
  val writeReqCnt = RegInit(0.U(32.W))

  // command fields
  val raddr_reg = RegInit(0.U(32.W))
  val waddr_reg = RegInit(0.U(32.W))
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val iter_reg  = RegInit(0.U(32.W))

  // -------------------------------
  // Default IO assignments
  // -------------------------------
  for (i <- 0 until inBW) {
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id     := 0.U
  }
  for (i <- 0 until outBW) {
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
    io.bankWrite(i).bank_id           := wbank_reg
    io.bankWrite(i).group_id      := 0.U
  }

  io.cmdReq.ready        := (state === idle)
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  // Always consume responses to avoid downstream backpressure deadlocks
  io.bankRead(0).io.resp.ready  := (state === compute)
  io.bankWrite(0).io.resp.ready := (state === compute)

  // -------------------------------
  // Helpers
  // -------------------------------
  def blockBase(sel: UInt): UInt = Mux(sel === 0.U, 0.U, InputNum.U)

  // "space available" means at least one block is not full
  val hasFreeBlock = !(blockFull(0) && blockFull(1))

  // -------------------------------
  // Main FSM
  // -------------------------------
  switch(state) {
    is(idle) {
      // reset runtime flags (not mandatory, but keeps things clean)
      when(io.cmdReq.fire) {
        state := compute

        raddr_reg := 0.U
        waddr_reg := 0.U
        rbank_reg := io.cmdReq.bits.cmd.op1_bank
        wbank_reg := io.cmdReq.bits.cmd.wr_bank
        iter_reg  := io.cmdReq.bits.cmd.iter

        readReqCnt  := 0.U
        readRespCnt := 0.U
        writeReqCnt := 0.U

        fillSel     := 0.U
        drainSel    := 0.U
        blockFull   := VecInit(Seq.fill(2)(false.B))
        fillRowIdx  := 0.U
        draining    := false.B
        drainColIdx := 0.U
      }
    }

    is(compute) {
      // ---------------------------
      // 1) READ REQ generation
      // Only issue a read when:
      // - still need reads (readReqCnt < iter_reg)
      // - there is space in buffers (not both blocks full)
      // - downstream ready handshake ok (via fire)
      // ---------------------------
      val wantRead = (readReqCnt < iter_reg) && hasFreeBlock
      io.bankRead(0).io.req.valid     := wantRead
      io.bankRead(0).io.req.bits.addr := raddr_reg + readReqCnt

      when(io.bankRead(0).io.req.fire) {
        readReqCnt := readReqCnt + 1.U
      }

      // ---------------------------
      // 2) READ RESP handling (fill ping-pong block)
      // Only advance fillRowIdx/blockFull on resp.fire
      // ---------------------------
      when(io.bankRead(0).io.resp.fire) {
        val dataWord = io.bankRead(0).io.resp.bits.data
        val base     = blockBase(fillSel)
        val rowIdx   = base + fillRowIdx

        for (col <- 0 until InputNum) {
          val hi = (col + 1) * inputWidth - 1
          val lo = col * inputWidth
          regArray(rowIdx)(col) := dataWord(hi, lo)
        }

        readRespCnt := readRespCnt + 1.U

        when(fillRowIdx === (InputNum - 1).U) {
          // this block becomes full
          blockFull(fillSel) := true.B
          fillRowIdx         := 0.U

          // switch to the other block if it is not full
          when(!blockFull(~fillSel)) {
            fillSel := ~fillSel
          }
        }.otherwise {
          fillRowIdx := fillRowIdx + 1.U
        }
      }

      // ---------------------------
      // 3) Start draining (writing) when not already draining
      // ---------------------------
      when(!draining) {
        when(blockFull(drainSel) && (writeReqCnt < iter_reg)) {
          draining    := true.B
          drainColIdx := 0.U
        }.elsewhen(blockFull(~drainSel) && (writeReqCnt < iter_reg)) {
          // if current drainSel block not full but the other is, swap
          drainSel    := ~drainSel
          draining    := true.B
          drainColIdx := 0.U
        }
      }

      // ---------------------------
      // 4) WRITE REQ generation (transpose)
      // Only advance drainColIdx / writeReqCnt on req.fire
      // ---------------------------
      when(draining) {
        val base = blockBase(drainSel)

        // compose one output row = one column of input block
        val col    = drainColIdx
        val packed = Cat((0 until InputNum).reverse.map { r =>
          regArray(base + r.U)(col)
        })

        val hasMoreWrite = (drainColIdx < InputNum.U) && (writeReqCnt < iter_reg)
        io.bankWrite(0).io.req.valid     := hasMoreWrite
        io.bankWrite(0).io.req.bits.addr := waddr_reg + writeReqCnt
        io.bankWrite(0).io.req.bits.data := packed
        io.bankWrite(0).io.req.bits.mask := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))

        when(io.bankWrite(0).io.req.fire) {
          writeReqCnt := writeReqCnt + 1.U

          when(drainColIdx === (InputNum - 1).U) {
            // finished draining this block
            draining            := false.B
            blockFull(drainSel) := false.B
            drainSel            := ~drainSel

            // move write base address by InputNum for next block’s outputs
            waddr_reg := waddr_reg + InputNum.U
            // move read base address too, purely optional; we already use readReqCnt for addr
            // raddr_reg := raddr_reg + InputNum.U
          }.otherwise {
            drainColIdx := drainColIdx + 1.U
          }
        }
      }

      // ---------------------------
      // 5) Completion condition (NO early complete)
      // Must ensure:
      // - all read req issued
      // - all read resp received
      // - all write req issued
      // - no blocks full, not draining
      // ---------------------------
      val done =
        (readReqCnt === iter_reg) &&
          (readRespCnt === iter_reg) &&
          (writeReqCnt === iter_reg) &&
          !blockFull(0) && !blockFull(1) &&
          !draining

      io.cmdResp.valid       := done
      io.cmdResp.bits.rob_id := rob_id_reg

      when(done && io.cmdResp.fire) {
        state := idle
      }
    }
  }

  io.status.idle    := (state === idle)
  io.status.running := (state === compute)
}
