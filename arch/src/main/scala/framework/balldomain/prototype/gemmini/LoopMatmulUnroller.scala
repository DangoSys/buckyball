package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import framework.top.GlobalConfig
import framework.balldomain.prototype.gemmini.configs.GemminiBallParam

/**
 * LoopMatmulUnroller — expands LOOP_WS into a stream of LoopCmd rows.
 *
 * FSM: sIdle → sAllocBanks → sPrimeLoad → sMainLoop (Row1/Row2) → sDrainLast → sFreeBanks → sDone
 *
 * Row layout per iteration:
 *   Row 1: [PRELOAD(cur), COMPUTE(cur)]
 *   Row 2: [MVOUT(cur), MVIN_A(next), MVIN_B(next)]
 * Last iteration: only Row 1, then sDrainLast handles final MVOUT.
 */
@instantiable
class LoopMatmulUnroller(val b: GlobalConfig) extends Module {
  val config    = GemminiBallParam()
  val DIM       = config.blockSize
  val elemSize  = config.inputWidth / 8
  val accBytes  = config.accWidth / 8
  val bankIdLen = log2Up(b.memDomain.bankNum)

  @public
  val io = IO(new Bundle {
    val start = Flipped(Decoupled(new LoopWsConfig(b)))
    val cmd   = Decoupled(new LoopCmd(b))
    val busy  = Output(Bool())
  })

  // FSM states
  val sIdle :: sAllocBanks :: sPrimeLoad :: sMainRow1 :: sMainRow2 :: sDrainLast :: sFreeBanks :: sDone :: Nil =
    Enum(8)
  val state                                                                                                    = RegInit(sIdle)

  // Config registers (latched at start)
  val cfg = Reg(new LoopWsConfig(b))

  // Loop iterator registers
  val i_reg = RegInit(0.U(16.W))
  val j_reg = RegInit(0.U(16.W))
  val k_reg = RegInit(0.U(16.W))

  // Total iterations and current iteration index
  val totalIter = cfg.max_i * cfg.max_j * cfg.max_k
  val curIter   = RegInit(0.U(32.W))

  // Whether this is the last iteration
  val isLastIter = curIter === (totalIter - 1.U)

  // Defaults
  io.start.ready := state === sIdle
  io.cmd.valid   := false.B
  io.cmd.bits    := 0.U.asTypeOf(new LoopCmd(b))
  io.busy        := state =/= sIdle

  // Helper to build an invalid LoopSubCmd slot
  def emptySlot(): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(b)))
    v.valid := false.B
    v.bits  := 0.U.asTypeOf(new LoopSubCmd(b))
    v
  }

  // Helper to build an MSET slot
  def msetSlot(
    bankId: UInt,
    alloc:  Boolean,
    row:    Int,
    col:    Int
  ): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(b)))
    v.valid         := true.B
    v.bits          := 0.U.asTypeOf(new LoopSubCmd(b))
    v.bits.cmdType  := (if (alloc) LoopSubCmdType.MSET_ALLOC else LoopSubCmdType.MSET_FREE)
    v.bits.bank_id  := bankId
    v.bits.bank_row := row.U
    v.bits.bank_col := col.U
    v
  }

  // Helper to build an MVIN slot
  def mvinSlot(bankId: UInt, addr: UInt, iter: UInt): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(b)))
    v.valid          := true.B
    v.bits           := 0.U.asTypeOf(new LoopSubCmd(b))
    v.bits.cmdType   := LoopSubCmdType.MVIN
    v.bits.bank_id   := bankId
    v.bits.dram_addr := addr
    v.bits.iter      := iter
    v
  }

  // Helper to build an MVOUT slot
  def mvoutSlot(bankId: UInt, addr: UInt, iter: UInt): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(b)))
    v.valid          := true.B
    v.bits           := 0.U.asTypeOf(new LoopSubCmd(b))
    v.bits.cmdType   := LoopSubCmdType.MVOUT
    v.bits.bank_id   := bankId
    v.bits.dram_addr := addr
    v.bits.iter      := iter
    v
  }

  // Address computation
  // addr_a(i,k) = dram_addr_a + i * stride_a + k * DIM * elemSize
  // addr_b(k,j) = dram_addr_b + k * stride_b + j * DIM * elemSize
  // addr_c(i,j) = dram_addr_c + i * stride_c + j * DIM * accBytes
  def addrA(i: UInt, k: UInt): UInt =
    cfg.dram_addr_a + i * cfg.stride_a + k * (DIM * elemSize).U

  def addrB(k: UInt, j: UInt): UInt =
    cfg.dram_addr_b + k * cfg.stride_b + j * (DIM * elemSize).U

  def addrC(i: UInt, j: UInt): UInt =
    cfg.dram_addr_c + i * cfg.stride_c + j * (DIM * accBytes).U

  // Next iterator values (advance k, then j, then i)
  val next_k = Wire(UInt(16.W))
  val next_j = Wire(UInt(16.W))
  val next_i = Wire(UInt(16.W))

  when(k_reg + 1.U < cfg.max_k) {
    next_k := k_reg + 1.U
    next_j := j_reg
    next_i := i_reg
  }.elsewhen(j_reg + 1.U < cfg.max_j) {
    next_k := 0.U
    next_j := j_reg + 1.U
    next_i := i_reg
  }.otherwise {
    next_k := 0.U
    next_j := 0.U
    next_i := i_reg + 1.U
  }

  // Compute mode: first k iteration uses PRELOADED (0), rest use ACCUMULATED (1)
  val computeMode = Mux(k_reg === 0.U, 0.U(2.W), 1.U(2.W))

  switch(state) {
    is(sIdle) {
      io.start.ready := true.B
      when(io.start.fire) {
        cfg     := io.start.bits
        i_reg   := 0.U
        j_reg   := 0.U
        k_reg   := 0.U
        curIter := 0.U
        state   := sAllocBanks
      }
    }

    // Row 0: [MSET_alloc(A), MSET_alloc(B), MSET_alloc(C), --]
    is(sAllocBanks) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := msetSlot(cfg.bank_a, alloc = true, row = 1, col = 1)
      io.cmd.bits.slots(1) := msetSlot(cfg.bank_b, alloc = true, row = 1, col = 1)
      io.cmd.bits.slots(2) := msetSlot(cfg.bank_c, alloc = true, row = 1, col = 4) // accWidth = 4x inputWidth
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sPrimeLoad
      }
    }

    // Row 1: [MVIN_A(0), MVIN_B(0), --, --]
    is(sPrimeLoad) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := mvinSlot(cfg.bank_a, addrA(0.U, 0.U), DIM.U)
      io.cmd.bits.slots(1) := mvinSlot(cfg.bank_b, addrB(0.U, 0.U), DIM.U)
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sMainRow1
      }
    }

    // Main loop Row 1: [PRELOAD(cur), COMPUTE(cur), --, --]
    is(sMainRow1) {
      io.cmd.valid := true.B
      val preSlot = Wire(Valid(new LoopSubCmd(b)))
      preSlot.valid         := true.B
      preSlot.bits          := 0.U.asTypeOf(new LoopSubCmd(b))
      preSlot.bits.cmdType  := LoopSubCmdType.PRELOAD
      preSlot.bits.op1_bank := cfg.bank_a // preload A (activations) into systolic array
      preSlot.bits.wr_bank  := cfg.bank_c
      preSlot.bits.iter     := DIM.U

      val compSlot = Wire(Valid(new LoopSubCmd(b)))
      compSlot.valid             := true.B
      compSlot.bits              := 0.U.asTypeOf(new LoopSubCmd(b))
      compSlot.bits.cmdType      := LoopSubCmdType.COMPUTE
      compSlot.bits.op1_bank     := cfg.bank_a
      compSlot.bits.op2_bank     := cfg.bank_b
      compSlot.bits.wr_bank      := cfg.bank_c
      compSlot.bits.compute_mode := computeMode
      compSlot.bits.iter         := DIM.U

      io.cmd.bits.slots(0) := preSlot
      io.cmd.bits.slots(1) := compSlot
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()

      when(io.cmd.fire) {
        when(isLastIter) {
          state := sDrainLast // Skip Row 2 for last iteration
        }.otherwise {
          state := sMainRow2
        }
      }
    }

    // Main loop Row 2: [MVOUT(cur), MVIN_A(next), MVIN_B(next), --]
    is(sMainRow2) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := mvoutSlot(cfg.bank_c, addrC(i_reg, j_reg), DIM.U)
      io.cmd.bits.slots(1) := mvinSlot(cfg.bank_a, addrA(next_i, next_k), DIM.U)
      io.cmd.bits.slots(2) := mvinSlot(cfg.bank_b, addrB(next_k, next_j), DIM.U)
      io.cmd.bits.slots(3) := emptySlot()

      when(io.cmd.fire) {
        // Advance iterators
        i_reg   := next_i
        j_reg   := next_j
        k_reg   := next_k
        curIter := curIter + 1.U
        state   := sMainRow1
      }
    }

    // Drain: emit final MVOUT for last iteration
    is(sDrainLast) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := mvoutSlot(cfg.bank_c, addrC(i_reg, j_reg), DIM.U)
      io.cmd.bits.slots(1) := emptySlot()
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sFreeBanks
      }
    }

    // Free: [MSET_free(A), MSET_free(B), MSET_free(C), --]
    is(sFreeBanks) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := msetSlot(cfg.bank_a, alloc = false, row = 0, col = 0)
      io.cmd.bits.slots(1) := msetSlot(cfg.bank_b, alloc = false, row = 0, col = 0)
      io.cmd.bits.slots(2) := msetSlot(cfg.bank_c, alloc = false, row = 0, col = 0)
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sDone
      }
    }

    is(sDone) {
      state := sIdle
    }
  }
}
