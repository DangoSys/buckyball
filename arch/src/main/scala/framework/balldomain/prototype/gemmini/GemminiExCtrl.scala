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
  val sIdle :: sPreloadReq :: sPreloadRead :: sPreloadFeed :: sComputeReq :: sComputeRead :: sComputeFeed :: sFlush :: sDrain :: sStore :: sCommit :: Nil =
    Enum(11)
  val state                                                                                                                                               = RegInit(sIdle)

  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))

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
  val read_row_cnt  = RegInit(0.U(log2Up(DIM + 1).W))
  val feed_row_cnt  = RegInit(0.U(log2Up(DIM + 1).W))
  val store_row_cnt = RegInit(0.U(log2Up(DIM + 1).W))
  val total_rows    = RegInit(DIM.U(log2Up(DIM + 1).W))
  val req_sent      = RegInit(false.B) // whether mesh.io.req has fired for this matmul

  // Sub-command from special field (live wire, only valid when cmdReq is valid)
  val sub_cmd = io.cmdReq.bits.cmd.special(3, 0)

  // Read response queues
  val rdQueue0 = Module(new Queue(new SramReadResp(b), entries = DIM))
  val rdQueue1 = Module(new Queue(new SramReadResp(b), entries = DIM))
  rdQueue0.io.enq <> io.bankReadResp(0)
  rdQueue1.io.enq <> io.bankReadResp(1)

  // Output buffer: collect mesh responses row by row
  val outBuf     = Reg(Vec(DIM, Vec(config.meshColumns, Vec(config.tileColumns, outputType))))
  val outBufRows = RegInit(0.U(log2Up(DIM + 1).W))

  // Per-port write completion tracking for sStore
  val port_written = RegInit(VecInit(Seq.fill(outBW)(false.B)))

  // =========================================================================
  // Defaults
  // =========================================================================
  io.cmdReq.ready := state === sIdle

  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

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
        rob_id_reg    := io.cmdReq.bits.rob_id
        op1_bank      := io.cmdReq.bits.cmd.op1_bank
        op2_bank      := io.cmdReq.bits.cmd.op2_bank
        wr_bank       := io.cmdReq.bits.cmd.wr_bank
        saved_rs1     := io.cmdReq.bits.cmd.rs1
        saved_rs2     := io.cmdReq.bits.cmd.rs2
        saved_sub_cmd := sub_cmd
        total_rows    := Mux(io.cmdReq.bits.cmd.iter === 0.U, DIM.U, io.cmdReq.bits.cmd.iter)

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
          read_row_cnt := 0.U
          feed_row_cnt := 0.U
          req_sent     := false.B
          state        := sPreloadRead
        }.elsewhen(sub_cmd === GemminiSubCmd.COMPUTE_PRELOADED || sub_cmd === GemminiSubCmd.COMPUTE_ACCUMULATED) {
          read_row_cnt := 0.U
          feed_row_cnt := 0.U
          outBufRows   := 0.U
          req_sent     := false.B
          state        := sComputeRead
        }.elsewhen(sub_cmd === GemminiSubCmd.FLUSH) {
          state := sFlush
        }
      }
    }

    // -----------------------------------------------------------------------
    // PRELOAD_READ: read D/B matrix from SRAM bank into rdQueue
    // Address starts from 0 (beginning of the bank allocation)
    // -----------------------------------------------------------------------
    is(sPreloadRead) {
      when(read_row_cnt < total_rows) {
        io.bankReadReq(0).valid     := true.B
        io.bankReadReq(0).bits.addr := read_row_cnt
        when(io.bankReadReq(0).ready) {
          read_row_cnt := read_row_cnt + 1.U
        }
      }.otherwise {
        state := sPreloadFeed
      }
    }

    // -----------------------------------------------------------------------
    // PRELOAD_FEED: send req to mesh, then feed D/B data row by row
    // For OS mode: preload bias/D into mesh via d input, a=0, b=0
    // For WS mode: preload weights via b input, a=0, d=0
    // -----------------------------------------------------------------------
    is(sPreloadFeed) {
      // Step 1: fire mesh.io.req (once)
      when(!req_sent) {
        mesh.io.req.valid                     := true.B
        mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
        mesh.io.req.bits.pe_control.propagate := ~cfg_dataflow // OS: propagate=1, WS: propagate=0 for preload
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
        when(rdQueue0.io.deq.valid) {
          val row_data = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))

          // a = 0 always for preload
          mesh.io.a.valid := true.B
          mesh.io.a.bits  := 0.U.asTypeOf(mesh.A_TYPE)

          when(cfg_dataflow === Dataflow.OS.id.U) {
            // OS: preload accumulator via d input
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := VecInit(row_data.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
          }.otherwise {
            // WS: preload weights via b input
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := VecInit(row_data.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)
          }

          when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
            rdQueue0.io.deq.ready := true.B
            feed_row_cnt          := feed_row_cnt + 1.U
          }
        }
      }

      // Step 3: all rows fed → commit
      when(req_sent && feed_row_cnt >= total_rows) {
        io.cmdResp.valid := true.B
        when(io.cmdResp.fire) {
          state := sIdle
        }
      }
    }

    // -----------------------------------------------------------------------
    // COMPUTE_READ: read A (port 0) and B/D (port 1) from SRAM banks
    // Both start from address 0
    // -----------------------------------------------------------------------
    is(sComputeRead) {
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

    // -----------------------------------------------------------------------
    // COMPUTE_FEED: send req to mesh, then feed A and B/D row by row
    // Collect mesh responses in parallel
    // -----------------------------------------------------------------------
    is(sComputeFeed) {
      // Always collect mesh output when valid
      when(mesh.io.resp.valid) {
        outBuf(outBufRows) := mesh.io.resp.bits.data
        outBufRows         := outBufRows + 1.U
      }

      // Step 1: fire mesh.io.req (once)
      when(!req_sent) {
        mesh.io.req.valid                     := true.B
        mesh.io.req.bits.pe_control.dataflow  := cfg_dataflow
        mesh.io.req.bits.pe_control.propagate := cfg_dataflow // OS: propagate=0, WS: propagate=1 for compute
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
        when(rdQueue0.io.deq.valid && rdQueue1.io.deq.valid) {
          val a_row  = rdQueue0.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))
          val bd_row = rdQueue1.io.deq.bits.data.asTypeOf(Vec(DIM, inputType))

          mesh.io.a.valid := true.B
          mesh.io.a.bits  := VecInit(a_row.grouped(config.tileRows).map(g => VecInit(g)).toSeq)

          when(cfg_dataflow === Dataflow.OS.id.U) {
            // OS: b carries multiply operand, d = 0 (accumulator already preloaded)
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := VecInit(bd_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := 0.U.asTypeOf(mesh.D_TYPE)
          }.otherwise {
            // WS: d carries input activations, b = 0
            mesh.io.b.valid := true.B
            mesh.io.b.bits  := 0.U.asTypeOf(mesh.B_TYPE)
            mesh.io.d.valid := true.B
            mesh.io.d.bits  := VecInit(bd_row.grouped(config.tileColumns).map(g => VecInit(g)).toSeq)
          }

          when(mesh.io.a.ready && mesh.io.b.ready && mesh.io.d.ready) {
            rdQueue0.io.deq.ready := true.B
            rdQueue1.io.deq.ready := true.B
            feed_row_cnt          := feed_row_cnt + 1.U
          }
        }
      }

      // Step 3: all rows fed → wait for all mesh responses
      when(req_sent && feed_row_cnt >= total_rows) {
        state := sDrain
      }
    }

    // -----------------------------------------------------------------------
    // DRAIN: wait for remaining mesh responses
    // -----------------------------------------------------------------------
    is(sDrain) {
      when(mesh.io.resp.valid) {
        outBuf(outBufRows) := mesh.io.resp.bits.data
        outBufRows         := outBufRows + 1.U
      }
      when(outBufRows >= total_rows) {
        store_row_cnt          := 0.U
        port_written.foreach(_ := false.B)
        state                  := sStore
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
        val row      = outBuf(store_row_cnt)
        val flat_row = VecInit(row.flatten)                // Vec of DIM SInt(accWidth)
        val row_bits = Cat(flat_row.reverse.map(_.asUInt)) // DIM * accWidth total bits

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
