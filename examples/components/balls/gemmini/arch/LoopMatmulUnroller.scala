package examples.balls.gemmini

import framework.acc.frontend.configs.FrontendParam
import framework.acc.memdomain.configs.MemDomainParam

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public}
import examples.balls.gemmini.configs.GemminiBallParam

/**
 * LoopMatmulUnroller — expands LOOP_WS into a stream of LoopCmd rows.
 *
 * FSM: sIdle → sAllocBanks → sPrimeLoad → sMainLoop (Row1/Row2) → sDrainLast → sFreeBanks → sDone
 *
 * Row layout per iteration:
 *   Row 1: [PRELOAD(cur), MVIN_B(cur)]
 *   Row 2: [COMPUTE(cur)]
 *   Row 3: [MVOUT(cur), MVIN_A(next)]
 * The final compute row is followed by a drain row and bank release.
 */
@instantiable
class LoopMatmulUnroller(val frontendParam: FrontendParam, val memParam: MemDomainParam, val ballParam: GemminiBallParam)
    extends Module {
  val config    = ballParam
  val DIM       = config.blockSize
  val elemSize  = config.inputWidth / 8
  val accBytes  = config.accWidth / 8
  val bankBytes = memParam.bankWidth / 8

  @public
  val io = IO(new Bundle {
    val start = Flipped(Decoupled(new LoopWsConfig(frontendParam, memParam, ballParam)))
    val cmd   = Decoupled(new LoopCmd(frontendParam, memParam, ballParam))
    val busy  = Output(Bool())
  })

  // FSM states
  val sIdle :: sAllocBanks :: sPrimeLoad :: sMainRow1 :: sCompute :: sMainRow2 :: sDrainLast :: sFreeBanks :: sDone :: Nil =
    Enum(9)
  val state                                                                                                                = RegInit(sIdle)

  // Config registers (latched at start)
  val cfg = Reg(new LoopWsConfig(frontendParam, memParam, ballParam))

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
  io.start.ready   := state === sIdle
  io.cmd.valid     := false.B
  io.cmd.bits      := 0.U.asTypeOf(new LoopCmd(frontendParam, memParam, ballParam))
  io.cmd.bits.last := state === sFreeBanks
  io.busy          := state =/= sIdle

  // Helper to build an invalid LoopSubCmd slot
  def emptySlot(): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(frontendParam, memParam, ballParam)))
    v.valid := false.B
    v.bits  := 0.U.asTypeOf(new LoopSubCmd(frontendParam, memParam, ballParam))
    v
  }

  // Helper to build an MSET slot
  def msetSlot(
    bankId: UInt,
    alloc:  Boolean,
    row:    Int,
    col:    Int
  ): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(frontendParam, memParam, ballParam)))
    v.valid         := true.B
    v.bits          := 0.U.asTypeOf(new LoopSubCmd(frontendParam, memParam, ballParam))
    v.bits.cmdType  := (if (alloc) LoopSubCmdType.MSET_ALLOC
                       else LoopSubCmdType.MSET_FREE)
    v.bits.bank_id  := bankId
    v.bits.bank_row := row.U
    v.bits.bank_col := col.U
    v
  }

  // Helper to build an MVIN slot
  def mvinSlot(
    bankId: UInt,
    addr:   UInt,
    iter:   UInt,
    stride: UInt
  ): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(frontendParam, memParam, ballParam)))
    v.valid          := true.B
    v.bits           := 0.U.asTypeOf(new LoopSubCmd(frontendParam, memParam, ballParam))
    v.bits.cmdType   := LoopSubCmdType.MVIN
    v.bits.bank_id   := bankId
    v.bits.dram_addr := addr
    v.bits.stride    := stride
    v.bits.iter      := iter
    v
  }

  // Helper to build an MVOUT slot
  def mvoutSlot(
    bankId: UInt,
    addr:   UInt,
    iter:   UInt,
    stride: UInt
  ): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(frontendParam, memParam, ballParam)))
    v.valid          := true.B
    v.bits           := 0.U.asTypeOf(new LoopSubCmd(frontendParam, memParam, ballParam))
    v.bits.cmdType   := LoopSubCmdType.MVOUT
    v.bits.bank_id   := bankId
    v.bits.dram_addr := addr
    v.bits.stride    := stride
    v.bits.iter      := iter
    v
  }

  // Address computation
  // addr_a(i,k) = dram_addr_a + i * stride_a + k * DIM * elemSize
  // addr_b(k,j) = dram_addr_b + k * stride_b + j * DIM * elemSize
  // addr_c(i,j) = dram_addr_c + i * stride_c + j * DIM * accBytes
  def addrA(i: UInt, k: UInt): UInt =
    cfg.dram_addr_a + i * DIM.U * cfg.stride_a + k * (DIM * elemSize).U

  def addrB(k: UInt, j: UInt): UInt =
    cfg.dram_addr_b + k * DIM.U * cfg.stride_b + j * (DIM * elemSize).U

  def addrC(i: UInt, j: UInt): UInt =
    cfg.dram_addr_c + i * DIM.U * cfg.stride_c + j * (DIM * accBytes).U

  def inMemStride(bytes: UInt): UInt =
    (bytes / bankBytes.U)(18, 0)

  def outMemStride(bytes: UInt): UInt =
    (bytes / (bankBytes * 4).U)(18, 0)

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
        assert(io.start.bits.max_i =/= 0.U && io.start.bits.max_j =/= 0.U && io.start.bits.max_k =/= 0.U)
        assert(
          io.start.bits.low_d && io.start.bits.dram_addr_d === 0.U,
          "Loop matmul requires zero initial accumulation"
        )
        assert(
          io.start.bits.bank_a =/= io.start.bits.bank_b && io.start.bits.bank_a =/= io.start.bits.bank_c && io.start.bits.bank_b =/= io.start.bits.bank_c
        )
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
      io.cmd.bits
        .slots(0)          := msetSlot(cfg.bank_a, alloc = true, row = 1, col = 1)
      io.cmd.bits
        .slots(1)          := msetSlot(cfg.bank_b, alloc = true, row = 1, col = 1)
      io.cmd.bits.slots(2) := msetSlot(
        cfg.bank_c,
        alloc = true,
        row = 1,
        col = 4
      ) // accWidth = 4x inputWidth
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sPrimeLoad
      }
    }

    // Load the activation core before preloading it.
    is(sPrimeLoad) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := mvinSlot(
        cfg.bank_a,
        addrA(0.U, 0.U),
        DIM.U,
        inMemStride(cfg.stride_a)
      )
      io.cmd.bits.slots(1) := emptySlot()
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sMainRow1
      }
    }

    // Preload activations while loading independent weights.
    is(sMainRow1) {
      io.cmd.valid := true.B
      val preSlot = Wire(Valid(new LoopSubCmd(frontendParam, memParam, ballParam)))
      preSlot.valid         := k_reg === 0.U
      preSlot.bits          := 0.U.asTypeOf(new LoopSubCmd(frontendParam, memParam, ballParam))
      preSlot.bits.cmdType  := LoopSubCmdType.PRELOAD
      preSlot.bits.op1_bank := cfg.bank_a // preload A (activations) into matrix array
      preSlot.bits.wr_bank  := cfg.bank_c
      preSlot.bits.iter     := DIM.U

      io.cmd.bits.slots(0)    := preSlot
      io.cmd.bits.slots(1)    := mvinSlot(
        cfg.bank_b,
        addrB(k_reg, j_reg),
        DIM.U,
        inMemStride(cfg.stride_b)
      )
      io.cmd.bits.slots(2)    := emptySlot()
      io.cmd.bits.slots(3)    := emptySlot()
      when(io.cmd.fire)(state := sCompute)
    }

    // Compute consumes the preceding preload and DMA row.
    is(sCompute) {
      io.cmd.valid := true.B
      val compSlot = Wire(Valid(new LoopSubCmd(frontendParam, memParam, ballParam)))
      compSlot.valid             := true.B
      compSlot.bits              := 0.U.asTypeOf(new LoopSubCmd(frontendParam, memParam, ballParam))
      compSlot.bits.cmdType      := LoopSubCmdType.COMPUTE
      compSlot.bits.op1_bank     := cfg.bank_a
      compSlot.bits.op2_bank     := cfg.bank_b
      compSlot.bits.wr_bank      := cfg.bank_c
      compSlot.bits.compute_mode := computeMode
      compSlot.bits.iter         := DIM.U

      io.cmd.bits.slots(0) := compSlot
      io.cmd.bits.slots(1) := emptySlot()
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()

      when(io.cmd.fire) {
        when(isLastIter) {
          state := sDrainLast // Drain after the final compute.
        }.otherwise {
          state := sMainRow2
        }
      }
    }

    // Store the result while preparing the next activation core.
    is(sMainRow2) {
      io.cmd.valid               := true.B
      io.cmd.bits.slots(0)       := mvoutSlot(
        cfg.bank_c,
        addrC(i_reg, j_reg),
        DIM.U,
        outMemStride(cfg.stride_c)
      )
      io.cmd.bits.slots(0).valid := k_reg + 1.U === cfg.max_k
      io.cmd.bits.slots(1)       := mvinSlot(
        cfg.bank_a,
        addrA(next_i, next_k),
        DIM.U,
        inMemStride(cfg.stride_a)
      )
      io.cmd.bits.slots(2)       := emptySlot()
      io.cmd.bits.slots(3)       := emptySlot()

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
      io.cmd.bits.slots(0) := mvoutSlot(
        cfg.bank_c,
        addrC(i_reg, j_reg),
        DIM.U,
        outMemStride(cfg.stride_c)
      )
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
      io.cmd.bits
        .slots(0)          := msetSlot(cfg.bank_a, alloc = false, row = 0, col = 0)
      io.cmd.bits
        .slots(1)          := msetSlot(cfg.bank_b, alloc = false, row = 0, col = 0)
      io.cmd.bits
        .slots(2)          := msetSlot(cfg.bank_c, alloc = false, row = 0, col = 0)
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
