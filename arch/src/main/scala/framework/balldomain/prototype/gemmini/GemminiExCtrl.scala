package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.memdomain.backend.banks.{SramReadReq, SramReadResp, SramWriteIO}
import framework.top.GlobalConfig
import framework.balldomain.prototype.gemmini.configs.GemminiBallParam
import gemmini._
import gemmini.Util._

/** Minimal tag for MeshWithDelays that satisfies TagQueueTag */
class SimpleTag extends Bundle with TagQueueTag {
  val rob = UInt(8.W)
  override def make_this_garbage(dummy: Int = 0): Unit =
    rob := 0.U
}

/** Sub-command encoding within the special field */
object GemminiSubCmd {
  val CONFIG              = 0.U(4.W)
  val PRELOAD             = 1.U(4.W)
  val COMPUTE_PRELOADED   = 2.U(4.W)
  val COMPUTE_ACCUMULATED = 3.U(4.W)
  val FLUSH               = 4.U(4.W)
}

/**
 * GemminiExCtrl — Execute controller adapter for MeshWithDelays.
 *
 * Receives BallRsIssue commands (config / preload / compute / flush),
 * drives gemmini.MeshWithDelays, reads data from SRAM banks, and
 * writes results back to SRAM banks.
 */
@instantiable
class GemminiExCtrl(val b: GlobalConfig) extends Module {
  val config = GemminiBallParam()
  val DIM    = config.blockSize // e.g., 16

  val ballMapping = b.ballDomain.ballIdMappings.find(_.ballName == "GemminiBall")
    .getOrElse(throw new IllegalArgumentException("GemminiBall not found in config"))
  val inBW        = ballMapping.inBW
  val outBW       = ballMapping.outBW

  val inputType  = SInt(config.inputWidth.W)
  val accType    = SInt(config.accWidth.W)
  val outputType = SInt(config.accWidth.W)

  @public
  val io = IO(new Bundle {
    val cmdReq       = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp      = Decoupled(new BallRsComplete(b))
    val bankReadReq  = Vec(inBW, Decoupled(new SramReadReq(b)))
    val bankReadResp = Vec(inBW, Flipped(Decoupled(new SramReadResp(b))))
    val bankWrite    = Vec(outBW, Flipped(new SramWriteIO(b)))
    val op1_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val op2_bank_o   = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val wr_bank_o    = Output(UInt(log2Up(b.memDomain.bankNum).W))
    val status       = new BallStatus
  })

  // =========================================================================
  // Instantiate Gemmini's MeshWithDelays (the real systolic array)
  // =========================================================================
  val mesh = Module(new MeshWithDelays(
    inputType = inputType,
    outputType = outputType,
    accType = accType,
    tagType = new SimpleTag,
    df = Dataflow.BOTH,
    tree_reduction = false,
    tile_latency = config.tileLatency,
    output_delay = config.outputDelay,
    tileRows = config.tileRows,
    tileColumns = config.tileColumns,
    meshRows = config.meshRows,
    meshColumns = config.meshColumns,
    leftBanks = 1,
    upBanks = 1,
    n_simultaneous_matmuls = 5
  ))

  // =========================================================================
  // Configuration registers (updated by CONFIG sub-command)
  // =========================================================================
  val cfg_dataflow    = RegInit(0.U(1.W)) // 0=OS, 1=WS
  val cfg_activation  = RegInit(0.U(2.W)) // 0=none, 1=relu
  val cfg_a_transpose = RegInit(false.B)
  val cfg_b_transpose = RegInit(false.B)
  val cfg_in_shift    = RegInit(0.U(log2Up(config.accWidth).W))

  // =========================================================================
  // State machine
  // =========================================================================
  val sIdle :: sPreloadReq :: sPreloadRead :: sPreloadFeed :: sComputeReq :: sComputeRead :: sComputeFeed :: sComputeFlush :: sFlush :: sDrain :: sStore :: sCommit :: Nil =
    Enum(12)
  val state                                                                                                                                                                = RegInit(sIdle)

  val rob_id_reg     = RegInit(0.U(log2Up(b.frontend.rob_entries).W))
  val is_sub_reg     = RegInit(false.B)
  val sub_rob_id_reg = RegInit(0.U(log2Up(b.frontend.sub_rob_depth * 4).W))

  // Saved rs1/rs2 from the command (latched at sIdle when cmdReq fires)
  val saved_rs1     = Reg(UInt(64.W))
  val saved_rs2     = Reg(UInt(64.W))
  val saved_sub_cmd = Reg(UInt(4.W))

  // Bank addressing
  val op1_bank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val op2_bank = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wr_bank  = RegInit(0.U(log2Up(b.memDomain.bankNum).W))

  io.op1_bank_o := op1_bank
  io.op2_bank_o := op2_bank
  io.wr_bank_o  := wr_bank

  // Iteration counters
  val read_row_cnt    = RegInit(0.U(log2Up(DIM + 1).W))
  val feed_row_cnt    = RegInit(0.U(log2Up(DIM + 1).W))
  val skip_rows_cnt   = RegInit(0.U(log2Up(DIM + 1).W)) // OS mode: count COMPUTE resp rows to discard
  val store_row_cnt   = RegInit(0.U(log2Up(DIM + 1).W))
  val total_rows      = RegInit(DIM.U(log2Up(DIM + 1).W))
  val req_sent        = RegInit(false.B)                // whether mesh.io.req has fired for this matmul
  val preload_pad_cnt = RegInit(0.U(log2Up(DIM + 1).W)) // WS+b_transpose: extra transposer drain rows

  // Sub-command from special field (live wire, only valid when cmdReq is valid)
  val sub_cmd = io.cmdReq.bits.cmd.special(3, 0)

  // Read response queues
  val rdQueue0 = Module(new Queue(new SramReadResp(b), entries = DIM))
  val rdQueue1 = Module(new Queue(new SramReadResp(b), entries = DIM))
  rdQueue0.io.enq <> io.bankReadResp(0)
  rdQueue1.io.enq <> io.bankReadResp(1)

  // Output buffer: collect mesh responses row by row
  val outBuf          = Reg(Vec(DIM, Vec(config.meshColumns, Vec(config.tileColumns, outputType))))
  val outBufRows      = RegInit(0.U(log2Up(DIM + 1).W))
  // OS mode: mesh outputs C rows in REVERSE order (last tile row first).
  // outBufCollected counts how many resp rows have been collected (0..total_rows)
  val outBufCollected = RegInit(0.U(log2Up(DIM + 1).W))

  // Per-port write completion tracking for sStore
  val port_written = RegInit(VecInit(Seq.fill(outBW)(false.B)))

  // =========================================================================
  // Defaults
  // =========================================================================
  io.cmdReq.ready := state === sIdle

  io.cmdResp.valid           := false.B
  io.cmdResp.bits.rob_id     := rob_id_reg
  io.cmdResp.bits.is_sub     := is_sub_reg
  io.cmdResp.bits.sub_rob_id := sub_rob_id_reg

  for (i <- 0 until inBW) {
    io.bankReadReq(i).valid     := false.B
    io.bankReadReq(i).bits.addr := 0.U
  }

  rdQueue0.io.deq.ready := false.B
  rdQueue1.io.deq.ready := false.B

  mesh.io.a.valid   := false.B
  mesh.io.a.bits    := 0.U.asTypeOf(mesh.A_TYPE)
  mesh.io.b.valid   := false.B
  mesh.io.b.bits    := 0.U.asTypeOf(mesh.B_TYPE)
  mesh.io.d.valid   := false.B
  mesh.io.d.bits    := 0.U.asTypeOf(mesh.D_TYPE)
  mesh.io.req.valid := false.B
  mesh.io.req.bits  := 0.U.asTypeOf(mesh.io.req.bits)

  io.bankWrite.foreach { bw =>
    bw.req.valid      := false.B
    bw.req.bits.addr  := 0.U
    bw.req.bits.data  := 0.U
    bw.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(false.B))
    bw.req.bits.wmode := true.B
    bw.resp.ready     := true.B
  }

  // =========================================================================
  // State Machine
  // =========================================================================
  switch(state) {

    // -----------------------------------------------------------------------
    // IDLE: accept new command, latch rs1/rs2/sub_cmd
    // -----------------------------------------------------------------------
    is(sIdle) {
      io.cmdReq.ready := true.B
      when(io.cmdReq.fire) {
        rob_id_reg     := io.cmdReq.bits.rob_id
        is_sub_reg     := io.cmdReq.bits.is_sub
        sub_rob_id_reg := io.cmdReq.bits.sub_rob_id
        op1_bank       := io.cmdReq.bits.cmd.op1_bank
        op2_bank       := io.cmdReq.bits.cmd.op2_bank
        wr_bank        := io.cmdReq.bits.cmd.wr_bank
        saved_rs1      := io.cmdReq.bits.cmd.rs1
        saved_rs2      := io.cmdReq.bits.cmd.rs2
        saved_sub_cmd  := sub_cmd
        total_rows     := Mux(io.cmdReq.bits.cmd.iter === 0.U, DIM.U, io.cmdReq.bits.cmd.iter)

        when(sub_cmd === GemminiSubCmd.CONFIG) {
          // CONFIG: decode from special field (rs2 with sub_cmd in [3:0])
          // Config params start at bit 4: [4]=dataflow, [6:5]=activation, [7]=a_transpose, [8]=b_transpose,
          // [40:9]=in_shift
          cfg_dataflow     := io.cmdReq.bits.cmd.special(4)
          cfg_activation   := io.cmdReq.bits.cmd.special(6, 5)
          cfg_a_transpose  := io.cmdReq.bits.cmd.special(7)
          cfg_b_transpose  := io.cmdReq.bits.cmd.special(8)
          cfg_in_shift     := io.cmdReq.bits.cmd.special(log2Up(config.accWidth) + 8, 9)
          // Commit immediately
          io.cmdResp.valid := true.B
          // Stay in sIdle if resp fires, else we need a config state
          // Since cmdReq.fire and cmdResp can both happen in same cycle:
          state            := sCommit
        }.elsewhen(sub_cmd === GemminiSubCmd.PRELOAD) {
          read_row_cnt    := 0.U
          feed_row_cnt    := 0.U
          preload_pad_cnt := 0.U
          req_sent        := false.B
          state           := sPreloadRead
        }.elsewhen(sub_cmd === GemminiSubCmd.COMPUTE_PRELOADED || sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED) {
          read_row_cnt    := 0.U
          feed_row_cnt    := 0.U
          skip_rows_cnt   := 0.U
          outBufRows      := 0.U
          outBufCollected := 0.U
          req_sent        := false.B
          state           := sComputeRead
        }.elsewhen(sub_cmd === GemminiSubCmd.FLUSH) {
          state := sFlush
        }
      }
    }

    // -----------------------------------------------------------------------
    // PRELOAD_READ: read A matrix (OS mode) or B/D weights (WS mode) from SRAM
    // OS mode: read A from op1_bank into rdQueue0, then send A through transposer
    //          in sPreloadFeed to pre-fill the AlwaysOutTransposer for later use
    //          in COMPUTE. The transposer accumulates A during PRELOAD, then
    //          naturally outputs A rows during COMPUTE (after dir switch).
    // WS mode: read weights (B/D) from op1_bank into rdQueue0 in REVERSE row order.
    //   MeshWithDelays uses a shift-chain for d: d[0] enters PE[0] first, then
    //   propagates down via out_c. After DIM PROPAGATE cycles, PE[r].c1 = d[DIM-1-r].
    //   To get PE[r].c1 = B[r][c] (correct WS weight layout), we must feed d in
    //   reverse: d[0]=B[DIM-1], d[1]=B[DIM-2], ..., d[DIM-1]=B[0].
    //   Reading SRAM in reverse (addr DIM-1 down to 0) achieves this.
    // -----------------------------------------------------------------------
    is(sPreloadRead) {
      when(cfg_dataflow === Dataflow.OS.id.U) {
        // OS mode: read A in forward order
        when(read_row_cnt < total_rows) {
          io.bankReadReq(0).valid     := true.B
          io.bankReadReq(0).bits.addr := read_row_cnt
          when(io.bankReadReq(0).ready) {
            read_row_cnt := read_row_cnt + 1.U
          }
        }.otherwise {
          state := sPreloadFeed
        }
      }.otherwise {
        // WS mode: read weights. For WS+b_transpose we feed B rows 0..DIM-1 so the
        // transposer outputs B's columns in order (col 0 = B[0][0],B[1][0],..). For WS
        // without transpose, reverse row order is used for c1 <- d mapping.
        when(read_row_cnt < total_rows) {
          io.bankReadReq(0).valid     := true.B
          io.bankReadReq(0).bits.addr := Mux(cfg_b_transpose, read_row_cnt, total_rows - 1.U - read_row_cnt)
          when(io.bankReadReq(0).ready) {
            read_row_cnt := read_row_cnt + 1.U
          }
        }.otherwise {
          state := sPreloadFeed
        }
      }
    }

    // -----------------------------------------------------------------------
    // PRELOAD_FEED: send req to mesh, then feed data row by row
    // For OS mode: preload A through transposer (a=A, b=0, d=0); propagate=1
    //   in_prop XOR: 0 XOR 1 = 1 → PE prop=1 (PROPAGATE): c1=d=0, outputs old c1
    //   After DIM rows of A, AlwaysOutTransposer dir flips; during COMPUTE,
    //   transposer outputs the A rows (as needed by systolic array).
    // For WS mode: preload weights via b input (a=0, b=B, d=0); propagate=0
    //   in_prop XOR: 0 XOR 0 = 0 → PE prop=0 (COMPUTE): c1 stays, b accumulated
    // -----------------------------------------------------------------------
    is(sPreloadFeed) {
      val wsNeedTransposeDrain = cfg_dataflow =/= Dataflow.OS.id.U && cfg_b_transpose
      val preload_rows         = Wire(UInt(mesh.io.req.bits.total_rows.getWidth.W))
      preload_rows := total_rows
      when(wsNeedTransposeDrain) {
        preload_rows := total_rows +& total_rows
      }
      val preload_flush = 0.U

      // Step 1: fire mesh.io.req (once)
      when(!req_sent) {
        mesh.io.req.valid                     := true.B
        mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
        mesh.io.req.bits.pe_control.propagate := 1.U // OS: propagate=1 (PROPAGATE, preload A into transposer); WS: propagate=1 (PROPAGATE, c1=d stores weight)
        mesh.io.req.bits.pe_control.shift     := cfg_in_shift
        mesh.io.req.bits.a_transpose          := cfg_a_transpose
        mesh.io.req.bits.bd_transpose         := cfg_b_transpose
        mesh.io.req.bits.total_rows           := preload_rows
        mesh.io.req.bits.tag.rob              := rob_id_reg
        mesh.io.req.bits.flush                := preload_flush
        when(mesh.io.req.fire) {
          req_sent := true.B
        }
      }

      // Step 2: feed data rows (only after req has fired)
      when(req_sent && feed_row_cnt < total_rows) {
        when(rdQueue0.io.deq.valid) {
          val row_data = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
          when(cfg_dataflow === Dataflow.OS.id.U) {
            // OS mode: feed A matrix through mesh.io.a → AlwaysOutTransposer
            // This pre-fills the transposer (DIM rows → dir flips to UP mode)
            // During COMPUTE, transposer outputs A rows from the top
            mesh.io.a.valid := true.B
            mesh.io.a.bits  := VecInit(row_data.grouped(config.tileRows).map(g => VecInit(g)).toSeq)
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)
          }.otherwise {
            // WS mode: preload weights via d input (PE WS PROPAGATE: c1 = d)
            mesh.io.a.valid := true.B
            mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := VecInit(row_data.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
          }
          when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
            rdQueue0.io.deq.ready := true.B
            feed_row_cnt          := feed_row_cnt + 1.U
          }
        }
      }.elsewhen(req_sent && wsNeedTransposeDrain && preload_pad_cnt < total_rows) {
        mesh.io.a.valid := true.B
        mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)
        mesh.io.b.valid := true.B
        mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
        mesh.io.d.valid := true.B
        mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)
        when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
          preload_pad_cnt := preload_pad_cnt + 1.U
        }
      }

      // Step 3: all rows fed → commit
      when(req_sent && feed_row_cnt >= total_rows) {
        when(!wsNeedTransposeDrain || preload_pad_cnt >= total_rows) {
          io.cmdResp.valid := true.B
          when(io.cmdResp.fire) {
            state := sIdle
          }
        }
      }
    }

    // -----------------------------------------------------------------------
    // COMPUTE_READ: read operands from SRAM banks
    // OS mode: only read B matrix from op2_bank (port 1)
    //          A matrix was already sent to transposer during PRELOAD
    // WS mode: read A from op1_bank (port 0) and D from op2_bank (port 1)
    // -----------------------------------------------------------------------
    is(sComputeRead) {
      when(cfg_dataflow === Dataflow.OS.id.U) {
        // OS mode: only read B from op2_bank
        when(read_row_cnt < total_rows) {
          io.bankReadReq(1).valid     := true.B
          io.bankReadReq(1).bits.addr := read_row_cnt
          when(io.bankReadReq(1).ready) {
            read_row_cnt := read_row_cnt + 1.U
          }
        }.otherwise {
          state := sComputeFeed
        }
      }.otherwise {
        // WS mode: read A from op1_bank (port 0) and D from op2_bank (port 1)
        when(read_row_cnt < total_rows) {
          io.bankReadReq(0).valid     := true.B
          io.bankReadReq(0).bits.addr := read_row_cnt
          io.bankReadReq(1).valid     := true.B
          io.bankReadReq(1).bits.addr := read_row_cnt

          when(io.bankReadReq(0).ready && io.bankReadReq(1).ready) {
            read_row_cnt := read_row_cnt + 1.U
          }
        }.otherwise {
          state := sComputeFeed
        }
      }
    }

    // -----------------------------------------------------------------------
    // COMPUTE_FEED: send req to mesh, then feed A and B/D row by row
    // OS mode: propagate=1 → in_prop XOR: 1 XOR 1 = 0 → PE prop=0 (COMPUTE)
    //   PE accumulates A*B into c1, outputs c2 (old/zero, discarded)
    //   A comes from AlwaysOutTransposer (pre-filled during PRELOAD); send a=0
    //   B comes from rdQueue1 (op2_bank)
    // WS mode: propagate=1 → in_prop XOR: 0 XOR 1 = 1 → PE prop=1 (PROPAGATE)
    //   PE outputs c1 (weights * activations), b passes through
    //   A comes from rdQueue0 (op1_bank), D from rdQueue1 (op2_bank)
    // -----------------------------------------------------------------------
    is(sComputeFeed) {
      // Collect/skip mesh output during COMPUTE feed phase.
      // WS mode: PRELOAD residual resp may still be trickling out here (the last few
      //   resp from PRELOAD's PROPAGATE phase overlap with the start of sComputeFeed).
      //   Discard all resp seen here; count them in skip_rows_cnt so that sDrain
      //   knows how many PRELOAD resp have already been flushed.
      // OS mode: COMPUTE resp (c2=0, garbage); discard and count in skip_rows_cnt.
      when(mesh.io.resp.valid) {
        skip_rows_cnt := skip_rows_cnt + 1.U
      }

      // Step 1: fire mesh.io.req (once)
      when(!req_sent) {
        mesh.io.req.valid                     := true.B
        mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
        // MeshWithDelays applies req.propagate with XOR semantics on in_prop.
        // After PRELOAD (req.propagate=1), WS compute must also send 1 to toggle
        // effective PE mode from PROPAGATE -> COMPUTE.
        mesh.io.req.bits.pe_control.propagate := 1.U
        mesh.io.req.bits.pe_control.shift     := cfg_in_shift
        mesh.io.req.bits.a_transpose          := cfg_a_transpose
        mesh.io.req.bits.bd_transpose         := cfg_b_transpose
        mesh.io.req.bits.total_rows           := total_rows
        mesh.io.req.bits.tag.rob              := rob_id_reg
        mesh.io.req.bits.flush                := 0.U
        when(mesh.io.req.fire) {
          req_sent := true.B
        }
      }

      // Step 2: feed data rows (only after req has fired)
      when(req_sent && feed_row_cnt < total_rows) {
        when(cfg_dataflow === Dataflow.OS.id.U) {
          // OS mode: A from transposer (pre-filled during PRELOAD); send a=0 to mesh.io.a
          // B from rdQueue1 (op2_bank)
          when(rdQueue1.io.deq.valid) {
            val b_row = rdQueue1.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
            // a=0: transposer already outputs A rows (accumulated during PRELOAD phase dir=LEFT)
            // After PRELOAD's DIM rows, transposer dir flips to UP, outputting A rows
            mesh.io.a.valid := true.B
            mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE) // a=0: let transposer output PRELOAD A
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := VecInit(b_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE) // d=0: don't overwrite c1
            when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
              rdQueue1.io.deq.ready := true.B
              feed_row_cnt          := feed_row_cnt + 1.U
            }
          }
        }.otherwise {
          // WS mode: activations from rdQueue0 via mesh.io.a, bias from rdQueue1 via mesh.io.b, d=0.
          // Keep normal 3-way handshake to avoid mesh backpressure lockup.
          when(rdQueue0.io.deq.valid && rdQueue1.io.deq.valid) {
            val act_row  = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
            val bias_row = rdQueue1.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))

            mesh.io.a.valid := true.B
            mesh.io.a.bits  := VecInit(act_row.grouped(config.tileRows).map(g => VecInit(g)).toSeq)
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := VecInit(bias_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)

            when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
              rdQueue0.io.deq.ready := true.B
              rdQueue1.io.deq.ready := true.B
              feed_row_cnt          := feed_row_cnt + 1.U
            }
          }
        }
      }

      // Step 3: all rows fed; OS mode waits until COMPUTE responses are discarded
      when(req_sent && feed_row_cnt >= total_rows) {
        when(cfg_dataflow === Dataflow.OS.id.U) {
          // OS mode: wait until all COMPUTE resp (c2=0, garbage) have been discarded
          when(skip_rows_cnt >= total_rows) {
            // OS mode: mesh outputs C rows in REVERSE order (last tile row first).
            // Initialize outBufRows to total_rows-1 so first resp → outBuf[total_rows-1],
            // last resp → outBuf[0], reversing back to correct row order.
            outBufRows      := total_rows - 1.U
            outBufCollected := 0.U
            req_sent        := false.B
            state           := sComputeFlush
          }
        }.otherwise {
          // WS: COMPUTE resp will appear in sDrain after the pipeline drains.
          // skip_rows_cnt already tracks how many PRELOAD resp have been seen here;
          // sDrain will skip the remaining PRELOAD resp (total_rows - skip_rows_cnt).
          outBufRows := 0.U
          state      := sDrain
        }
      }
    }

    // -----------------------------------------------------------------------
    // COMPUTE_FLUSH (OS mode only): issue flush req with propagate=1
    // in_prop XOR: 0 XOR 1 = 1 → PE prop=1 (PROPAGATE) → outputs c1 (A*B result)
    // Feed zeros to advance rows through the pipeline
    // -----------------------------------------------------------------------
    is(sComputeFlush) {
      // Collect mesh output (this is the real A*B result from c1)
      // OS PROPAGATE outputs rows in reverse: last tile row first, first tile row last.
      // Count down outBufRows so outBuf[total_rows-1] = first resp, outBuf[0] = last resp.
      when(mesh.io.resp.valid) {
        outBuf(outBufRows) := mesh.io.resp.bits.data
        outBufRows         := outBufRows - 1.U
        outBufCollected    := outBufCollected + 1.U
      }

      when(!req_sent) {
        mesh.io.req.valid                     := true.B
        mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
        mesh.io.req.bits.pe_control.propagate := 1.U // XOR: 0→1, PE prop=1, outputs c1
        mesh.io.req.bits.pe_control.shift     := cfg_in_shift
        mesh.io.req.bits.a_transpose          := cfg_a_transpose
        mesh.io.req.bits.bd_transpose         := cfg_b_transpose
        mesh.io.req.bits.total_rows           := total_rows
        mesh.io.req.bits.tag.rob              := rob_id_reg
        mesh.io.req.bits.flush                := 0.U
        when(mesh.io.req.fire) {
          req_sent     := true.B
          feed_row_cnt := 0.U
        }
      }

      // Feed zeros to advance pipeline
      when(req_sent && feed_row_cnt < total_rows) {
        mesh.io.a.valid := true.B
        mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)
        mesh.io.b.valid := true.B
        mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
        mesh.io.d.valid := true.B
        mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)
        when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
          feed_row_cnt := feed_row_cnt + 1.U
        }
      }

      // All flush rows fed → proceed to drain to collect any remaining responses
      when(req_sent && feed_row_cnt >= total_rows) {
        state := sDrain
      }
    }

    // -----------------------------------------------------------------------
    // DRAIN: wait for remaining mesh responses
    // -----------------------------------------------------------------------
    is(sDrain) {
      when(mesh.io.resp.valid) {
        when(cfg_dataflow === Dataflow.OS.id.U) {
          // OS mode: continue reverse collection
          outBuf(outBufRows) := mesh.io.resp.bits.data
          outBufRows         := outBufRows - 1.U
          outBufCollected    := outBufCollected + 1.U
        }.otherwise {
          // WS mode: collect responses directly.
          outBuf(outBufRows) := mesh.io.resp.bits.data
          outBufRows         := outBufRows + 1.U
        }
      }
      when(cfg_dataflow === Dataflow.OS.id.U) {
        when(outBufCollected >= total_rows) {
          store_row_cnt          := 0.U
          port_written.foreach(_ := false.B)
          state                  := sStore
        }
      }.otherwise {
        when(outBufRows >= total_rows) {
          store_row_cnt          := 0.U
          port_written.foreach(_ := false.B)
          state                  := sStore
        }
      }
    }

    // -----------------------------------------------------------------------
    // STORE: write results back to SRAM bank
    // Each row: DIM elements × accWidth bits = 512 bits
    // bankWidth = 128 bits, outBW = 4 write ports → 4×128 = 512 bits per cycle
    // Use per-port written flags to handle non-simultaneous ready signals
    // -----------------------------------------------------------------------
    is(sStore) {
      when(store_row_cnt < total_rows) {
        val row        = outBuf(store_row_cnt)
        // MeshWithDelays d path reverses element order within each write-port chunk.
        // For WS+b_transpose we reverse each chunk so stored order is col0..col15.
        val flat_raw   = VecInit(row.flatten)
        val tileW      = b.memDomain.bankWidth / config.accWidth
        val flat_order = VecInit((0 until DIM).map { i =>
          val t = i / tileW
          val j = t * tileW + (tileW - 1 - (i % tileW))
          Mux(cfg_dataflow =/= Dataflow.OS.id.U && cfg_b_transpose, flat_raw(j), flat_raw(i))
        })
        val flat_row   = VecInit(flat_order.map(x => (x >> cfg_in_shift).asSInt))
        val row_bits   = Cat(flat_row.reverse.map(_.asUInt))

        val bitsPerPort = b.memDomain.bankWidth

        for (i <- 0 until outBW) {
          when(!port_written(i)) {
            val slice = row_bits((i + 1) * bitsPerPort - 1, i * bitsPerPort)
            io.bankWrite(i).req.valid      := true.B
            io.bankWrite(i).req.bits.addr  := store_row_cnt
            io.bankWrite(i).req.bits.data  := slice
            io.bankWrite(i).req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
            io.bankWrite(i).req.bits.wmode := true.B
            when(io.bankWrite(i).req.ready) {
              port_written(i) := true.B
            }
          }
        }

        // All ports done for this row → advance to next row
        when(port_written.asUInt.andR) {
          store_row_cnt          := store_row_cnt + 1.U
          port_written.foreach(_ := false.B)
        }
      }.otherwise {
        state := sCommit
      }
    }

    // -----------------------------------------------------------------------
    // COMMIT: send completion response
    // -----------------------------------------------------------------------
    is(sCommit) {
      io.cmdResp.valid := true.B
      when(io.cmdResp.fire) {
        state := sIdle
      }
    }

    // -----------------------------------------------------------------------
    // FLUSH: send flush to mesh, commit
    // -----------------------------------------------------------------------
    is(sFlush) {
      mesh.io.req.valid           := true.B
      mesh.io.req.bits.flush      := 2.U
      mesh.io.req.bits.total_rows := DIM.U
      mesh.io.req.bits.tag.rob    := rob_id_reg

      mesh.io.a.valid := true.B
      mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)
      mesh.io.b.valid := true.B
      mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
      mesh.io.d.valid := true.B
      mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)

      when(mesh.io.req.ready) {
        io.cmdResp.valid := true.B
        when(io.cmdResp.fire) {
          state := sIdle
        }
      }
    }
  }

  // =========================================================================
  // Status
  // =========================================================================
  io.status.idle    := state === sIdle
  io.status.running := state =/= sIdle
}
