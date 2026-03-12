package framework.balldomain.prototype.trace

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.balldomain.rs.{BallRsComplete, BallRsIssue}
import framework.balldomain.blink.{BallStatus, BankRead, BankWrite}
import framework.memdomain.backend.banks.SramBank
import framework.top.GlobalConfig

/**
 * Trace — TraceBall inner processing unit.
 *
 * Handles two instruction types:
 *   - bdb_counter (funct7=48): cycle counter management (START/STOP/READ)
 *   - bdb_backdoor (funct7=49): SRAM backdoor read/write via DPI-C
 *
 * Backdoor write: C++ generates (row, data) via DPI-C each iteration,
 *   RTL writes to external bank at (wbank_reg, row).
 * Backdoor read: RTL reads external bank at (rbank_reg, row from DPI-C),
 *   sends data back to C++ via DPI-C for logging.
 *
 * bank_id always comes from the instruction encoding (rs1),
 * row and data come from DPI-C (C++ auto-increments row each call).
 */
@instantiable
class Trace(val b: GlobalConfig) extends Module {

  val ballMapping = b.ballDomain.ballIdMappings
    .find(_.ballName == "TraceBall")
    .getOrElse(throw new IllegalArgumentException("TraceBall not found in config"))

  val inBW  = ballMapping.inBW
  val outBW = ballMapping.outBW

  val bankWidth = b.memDomain.bankWidth

  @public
  val io = IO(new Bundle {
    val cmdReq    = Flipped(Decoupled(new BallRsIssue(b)))
    val cmdResp   = Decoupled(new BallRsComplete(b))
    val bankRead  = Vec(inBW, Flipped(new BankRead(b)))
    val bankWrite = Vec(outBW, Flipped(new BankWrite(b)))
    val status    = new BallStatus
  })

  // ============================================================
  // Constants
  // ============================================================
  val CTR_START = 0.U(4.W)
  val CTR_STOP  = 1.U(4.W)
  val CTR_READ  = 2.U(4.W)

  val NUM_COUNTERS = 16

  // ============================================================
  // State machine
  // ============================================================
  val idle :: sCounter :: sBdReadExt :: sBdReadExtResp :: sBdGetWriteData :: sBdWriteExt :: sBdWriteExtResp :: complete :: Nil =
    Enum(8)
  val state                                                                                                                    = RegInit(idle)

  // ============================================================
  // Registers
  // ============================================================
  val rob_id_reg = RegInit(0.U(log2Up(b.frontend.rob_entries).W))

  // Command decode registers
  val isRead_reg  = RegInit(false.B) // BB_RD0 flag
  val isWrite_reg = RegInit(false.B) // BB_WR flag
  val iter_reg    = RegInit(0.U(16.W))
  val iterCnt     = RegInit(0.U(16.W))

  // Counter-specific registers
  val subcmd_reg  = RegInit(0.U(4.W))
  val ctr_id_reg  = RegInit(0.U(4.W))
  val payload_reg = RegInit(0.U(56.W))

  // Cycle counters: 16 independent counters
  val cycleCounter = RegInit(0.U(64.W))
  cycleCounter := cycleCounter + 1.U

  val ctrStartCycle = RegInit(VecInit(Seq.fill(NUM_COUNTERS)(0.U(64.W))))
  val ctrTag        = RegInit(VecInit(Seq.fill(NUM_COUNTERS)(0.U(56.W))))
  val ctrActive     = RegInit(VecInit(Seq.fill(NUM_COUNTERS)(false.B)))

  // Bank metadata registers (from instruction encoding)
  val rbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))
  val wbank_reg = RegInit(0.U(log2Up(b.memDomain.bankNum).W))

  // Backdoor address/data (row from DPI-C, data from DPI-C for writes)
  val bd_addr_reg = RegInit(0.U(32.W))
  val bd_data_reg = RegInit(0.U(bankWidth.W))

  // ============================================================
  // Private SramBank (staging buffer for future use)
  // ============================================================
  val privBank = Module(new SramBank(b))

  // Default: private bank idle
  privBank.io.sramRead.req.valid       := false.B
  privBank.io.sramRead.req.bits.addr   := 0.U
  privBank.io.sramRead.resp.ready      := false.B
  privBank.io.sramWrite.req.valid      := false.B
  privBank.io.sramWrite.req.bits.addr  := 0.U
  privBank.io.sramWrite.req.bits.data  := 0.U
  privBank.io.sramWrite.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(true.B))
  privBank.io.sramWrite.req.bits.wmode := false.B
  privBank.io.sramWrite.resp.ready     := true.B

  // ============================================================
  // DPI-C modules
  // ============================================================
  val ctraceDpi = Module(new CTraceDPI)
  ctraceDpi.io.subcmd  := 0.U
  ctraceDpi.io.ctr_id  := 0.U
  ctraceDpi.io.tag     := 0.U
  ctraceDpi.io.elapsed := 0.U
  ctraceDpi.io.cycle   := cycleCounter
  ctraceDpi.io.enable  := false.B

  val bdGetReadAddr  = Module(new BackdoorGetReadAddrDPI)
  val bdGetWriteAddr = Module(new BackdoorGetWriteAddrDPI)
  val bdGetWriteData = Module(new BackdoorGetWriteDataDPI)
  val bdPutReadData  = Module(new BackdoorPutReadDataDPI)
  val bdPutWriteDone = Module(new BackdoorPutWriteDoneDPI)

  bdGetReadAddr.io.enable  := false.B
  bdGetWriteAddr.io.enable := false.B
  bdGetWriteData.io.enable := false.B

  bdPutReadData.io.bank_id := 0.U
  bdPutReadData.io.row     := 0.U
  bdPutReadData.io.data_lo := 0.U
  bdPutReadData.io.data_hi := 0.U
  bdPutReadData.io.enable  := false.B

  bdPutWriteDone.io.bank_id := 0.U
  bdPutWriteDone.io.row     := 0.U
  bdPutWriteDone.io.data_lo := 0.U
  bdPutWriteDone.io.data_hi := 0.U
  bdPutWriteDone.io.enable  := false.B

  // ============================================================
  // External bank port defaults
  // ============================================================
  for (i <- 0 until inBW) {
    io.bankRead(i).rob_id           := rob_id_reg
    io.bankRead(i).ball_id          := 0.U
    io.bankRead(i).bank_id          := rbank_reg
    io.bankRead(i).group_id         := 0.U
    io.bankRead(i).io.req.valid     := false.B
    io.bankRead(i).io.req.bits.addr := 0.U
    io.bankRead(i).io.resp.ready    := false.B
  }

  for (i <- 0 until outBW) {
    io.bankWrite(i).rob_id            := rob_id_reg
    io.bankWrite(i).ball_id           := 0.U
    io.bankWrite(i).bank_id           := wbank_reg
    io.bankWrite(i).group_id          := 0.U
    io.bankWrite(i).io.req.valid      := false.B
    io.bankWrite(i).io.req.bits.addr  := 0.U
    io.bankWrite(i).io.req.bits.data  := 0.U
    io.bankWrite(i).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(0.U(1.W)))
    io.bankWrite(i).io.req.bits.wmode := false.B
    io.bankWrite(i).io.resp.ready     := false.B
  }

  // ============================================================
  // Command interface
  // ============================================================
  io.cmdReq.ready        := state === idle
  io.cmdResp.valid       := false.B
  io.cmdResp.bits.rob_id := rob_id_reg

  // ============================================================
  // State machine
  // ============================================================
  switch(state) {
    is(idle) {
      when(io.cmdReq.fire) {
        rob_id_reg := io.cmdReq.bits.rob_id

        val cmd = io.cmdReq.bits.cmd
        val rs2 = cmd.rs2

        // Distinguish counter vs backdoor by op1_en / wr_spad_en
        val isBackdoor = cmd.op1_en || cmd.wr_spad_en

        isRead_reg  := cmd.op1_en     // BB_RD0
        isWrite_reg := cmd.wr_spad_en // BB_WR
        iter_reg    := cmd.iter
        iterCnt     := 0.U

        // Counter fields from rs2
        subcmd_reg  := rs2(3, 0)
        ctr_id_reg  := rs2(7, 4)
        payload_reg := rs2(63, 8)

        // Bank IDs from decoded cmd
        rbank_reg := cmd.op1_bank
        wbank_reg := cmd.wr_bank

        when(!isBackdoor) {
          state := sCounter
        }.elsewhen(cmd.wr_spad_en) {
          // Backdoor write: get row+data from DPI-C, write external bank
          state := sBdGetWriteData
        }.otherwise {
          // Backdoor read: get row from DPI-C, read external bank
          state := sBdReadExt
        }
      }
    }

    // ----------------------------------------------------------
    // Counter instruction: execute and complete in 1 cycle
    // ----------------------------------------------------------
    is(sCounter) {
      val cid = ctr_id_reg

      ctraceDpi.io.subcmd := subcmd_reg
      ctraceDpi.io.ctr_id := cid

      switch(subcmd_reg) {
        is(CTR_START) {
          ctrStartCycle(cid) := cycleCounter
          ctrTag(cid)        := payload_reg
          ctrActive(cid)     := true.B

          ctraceDpi.io.tag     := payload_reg
          ctraceDpi.io.elapsed := 0.U
          ctraceDpi.io.enable  := true.B
        }
        is(CTR_STOP) {
          val elapsed = cycleCounter - ctrStartCycle(cid)
          ctrActive(cid) := false.B

          ctraceDpi.io.tag     := ctrTag(cid)
          ctraceDpi.io.elapsed := elapsed
          ctraceDpi.io.enable  := true.B
        }
        is(CTR_READ) {
          val current = cycleCounter - ctrStartCycle(cid)

          ctraceDpi.io.tag     := ctrTag(cid)
          ctraceDpi.io.elapsed := current
          ctraceDpi.io.enable  := true.B
        }
      }

      state := complete
    }

    // ----------------------------------------------------------
    // Backdoor read: get row from DPI-C, read external bank
    // ----------------------------------------------------------
    is(sBdReadExt) {
      // Get row from DPI-C (C++ auto-increments)
      bdGetReadAddr.io.enable := true.B
      val row = bdGetReadAddr.io.result(31, 0)

      bd_addr_reg := row

      // Issue read to external bank (bank_id from instruction)
      io.bankRead(0).io.req.valid     := true.B
      io.bankRead(0).io.req.bits.addr := row
      io.bankRead(0).io.resp.ready    := true.B

      when(io.bankRead(0).io.req.fire) {
        state := sBdReadExtResp
      }
    }

    is(sBdReadExtResp) {
      io.bankRead(0).io.resp.ready := true.B

      when(io.bankRead(0).io.resp.valid) {
        val data = io.bankRead(0).io.resp.bits.data

        // Output data via DPI-C
        bdPutReadData.io.bank_id := rbank_reg
        bdPutReadData.io.row     := bd_addr_reg
        bdPutReadData.io.data_lo := data(63, 0)
        bdPutReadData.io.data_hi := data(127, 64)
        bdPutReadData.io.enable  := true.B

        // Check if more iterations
        iterCnt := iterCnt + 1.U
        when(iterCnt >= iter_reg) {
          state := complete
        }.otherwise {
          state := sBdReadExt
        }
      }
    }

    // ----------------------------------------------------------
    // Backdoor write: get row+data from DPI-C, write external bank
    // ----------------------------------------------------------
    is(sBdGetWriteData) {
      // Get row from DPI-C (C++ auto-increments and pre-generates data)
      bdGetWriteAddr.io.enable := true.B
      val row = bdGetWriteAddr.io.result(31, 0)
      bd_addr_reg := row

      // Get data from DPI-C
      bdGetWriteData.io.enable := true.B
      val fullData = Cat(bdGetWriteData.io.data_hi, bdGetWriteData.io.data_lo)
      bd_data_reg := fullData

      state := sBdWriteExt
    }

    is(sBdWriteExt) {
      // Write to external bank (bank_id from instruction)
      io.bankWrite(0).io.req.valid      := true.B
      io.bankWrite(0).io.req.bits.addr  := bd_addr_reg
      io.bankWrite(0).io.req.bits.data  := bd_data_reg
      io.bankWrite(0).io.req.bits.mask  := VecInit(Seq.fill(b.memDomain.bankMaskLen)(1.U(1.W)))
      io.bankWrite(0).io.req.bits.wmode := false.B
      io.bankWrite(0).io.resp.ready     := true.B

      when(io.bankWrite(0).io.req.fire) {
        state := sBdWriteExtResp
      }
    }

    is(sBdWriteExtResp) {
      io.bankWrite(0).io.resp.ready := true.B

      when(io.bankWrite(0).io.resp.valid) {
        // Log the write via DPI-C
        bdPutWriteDone.io.bank_id := wbank_reg
        bdPutWriteDone.io.row     := bd_addr_reg
        bdPutWriteDone.io.data_lo := bd_data_reg(63, 0)
        bdPutWriteDone.io.data_hi := bd_data_reg(127, 64)
        bdPutWriteDone.io.enable  := true.B

        // Check if more iterations
        iterCnt := iterCnt + 1.U
        when(iterCnt >= iter_reg) {
          state := complete
        }.otherwise {
          state := sBdGetWriteData
        }
      }
    }

    // ----------------------------------------------------------
    // Complete: fire cmdResp
    // ----------------------------------------------------------
    is(complete) {
      io.cmdResp.valid       := true.B
      io.cmdResp.bits.rob_id := rob_id_reg
      when(io.cmdResp.fire) {
        state := idle
      }
    }
  }

  // ============================================================
  // Status
  // ============================================================
  io.status.idle    := state === idle
  io.status.running := state =/= idle && state =/= complete
}
