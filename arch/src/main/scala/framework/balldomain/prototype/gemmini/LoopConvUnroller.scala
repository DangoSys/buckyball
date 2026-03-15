package framework.balldomain.prototype.gemmini

import chisel3._
import chisel3.util._
import chisel3.experimental.hierarchy.{instantiable, public, Instance, Instantiate}
import framework.top.GlobalConfig
import framework.balldomain.prototype.gemmini.configs.GemminiBallParam

/**
 * LoopConvUnroller — expands LOOP_CONV_WS into LoopCmd rows.
 *
 * Simplified iteration:
 *   batch → orow → ocol → och → krow → kcol → kch (innermost)
 *
 * Each (orow, ocol, och, krow, kcol, kch) tile generates:
 *   MVIN input + MVIN weight → PRELOAD + COMPUTE → (on last k-iter) MVOUT
 *
 * First implementation: no pooling, no im2col tiling, single DIM×DIM tiles.
 */
@instantiable
class LoopConvUnroller(val b: GlobalConfig) extends Module {
  val config   = GemminiBallParam()
  val DIM      = config.blockSize
  val elemSize = config.inputWidth / 8
  val accBytes = config.accWidth / 8

  @public
  val io = IO(new Bundle {
    val start = Flipped(Decoupled(new LoopConvWsConfig(b)))
    val cmd   = Decoupled(new LoopCmd(b))
    val busy  = Output(Bool())
  })

  val addrGen: Instance[LoopConvAddrGen] = Instantiate(new LoopConvAddrGen(b))

  // FSM states
  val sIdle :: sAlloc :: sMvinInput :: sMvinWeight :: sCompute :: sMvout :: sFree :: sDone :: Nil = Enum(8)
  val state                                                                                       = RegInit(sIdle)

  val cfg = Reg(new LoopConvWsConfig(b))

  // Iterator registers
  val batch_reg = RegInit(0.U(16.W))
  val orow_reg  = RegInit(0.U(16.W))
  val ocol_reg  = RegInit(0.U(16.W))
  val och_reg   = RegInit(0.U(16.W))
  val krow_reg  = RegInit(0.U(16.W))
  val kcol_reg  = RegInit(0.U(16.W))
  val kch_reg   = RegInit(0.U(16.W))

  // Is this the first k-iteration for this (orow, ocol, och)?
  val isFirstK = krow_reg === 0.U && kcol_reg === 0.U && kch_reg === 0.U

  // Is this the last k-iteration?
  val isLastK = (krow_reg === cfg.kernel_dim - 1.U) &&
    (kcol_reg === cfg.kernel_dim - 1.U) &&
    (kch_reg + DIM.U >= cfg.in_channels) // Tile boundary

  // Connect address generator
  addrGen.io.cfg   := cfg
  addrGen.io.batch := batch_reg
  addrGen.io.orow  := orow_reg
  addrGen.io.ocol  := ocol_reg
  addrGen.io.och   := och_reg
  addrGen.io.krow  := krow_reg
  addrGen.io.kcol  := kcol_reg
  addrGen.io.kch   := kch_reg

  // Defaults
  io.start.ready := state === sIdle
  io.cmd.valid   := false.B
  io.cmd.bits    := 0.U.asTypeOf(new LoopCmd(b))
  io.busy        := state =/= sIdle

  def emptySlot(): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(b)))
    v.valid := false.B
    v.bits  := 0.U.asTypeOf(new LoopSubCmd(b))
    v
  }

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

  def mvinSlot(bankId: UInt, addr: UInt): Valid[LoopSubCmd] = {
    val v = Wire(Valid(new LoopSubCmd(b)))
    v.valid          := true.B
    v.bits           := 0.U.asTypeOf(new LoopSubCmd(b))
    v.bits.cmdType   := LoopSubCmdType.MVIN
    v.bits.bank_id   := bankId
    v.bits.dram_addr := addr
    v.bits.iter      := DIM.U
    v
  }

  // Advance k-iterators (kch → kcol → krow), then output iterators (och → ocol → orow → batch)
  def advanceIter(): Unit = {
    when(kch_reg + DIM.U < cfg.in_channels) {
      kch_reg := kch_reg + DIM.U
    }.elsewhen(kcol_reg + 1.U < cfg.kernel_dim) {
      kch_reg := 0.U; kcol_reg := kcol_reg + 1.U
    }.elsewhen(krow_reg + 1.U < cfg.kernel_dim) {
      kch_reg := 0.U; kcol_reg := 0.U; krow_reg := krow_reg + 1.U
    }.elsewhen(och_reg + DIM.U < cfg.out_channels) {
      kch_reg := 0.U; kcol_reg := 0.U; krow_reg := 0.U; och_reg := och_reg + DIM.U
    }.elsewhen(ocol_reg + 1.U < cfg.out_dim) {
      kch_reg  := 0.U; kcol_reg := 0.U; krow_reg := 0.U; och_reg := 0.U
      ocol_reg := ocol_reg + 1.U
    }.elsewhen(orow_reg + 1.U < cfg.out_dim) {
      kch_reg  := 0.U; kcol_reg := 0.U; krow_reg := 0.U; och_reg := 0.U; ocol_reg := 0.U
      orow_reg := orow_reg + 1.U
    }.elsewhen(batch_reg + 1.U < cfg.batch_size) {
      kch_reg   := 0.U; kcol_reg := 0.U; krow_reg := 0.U; och_reg := 0.U; ocol_reg := 0.U; orow_reg := 0.U
      batch_reg := batch_reg + 1.U
    }.otherwise {
      state := sFree // All iterations done
    }
  }

  // Track whether all iterations are done (used to decide next state)
  val allDone = (batch_reg === cfg.batch_size - 1.U) &&
    (orow_reg === cfg.out_dim - 1.U) && (ocol_reg === cfg.out_dim - 1.U) &&
    (och_reg + DIM.U >= cfg.out_channels) && isLastK

  switch(state) {
    is(sIdle) {
      io.start.ready := true.B
      when(io.start.fire) {
        cfg       := io.start.bits
        batch_reg := 0.U; orow_reg := 0.U; ocol_reg := 0.U; och_reg := 0.U
        krow_reg  := 0.U; kcol_reg := 0.U; kch_reg  := 0.U
        state     := sAlloc
      }
    }

    // [MSET_alloc(input), MSET_alloc(weight), MSET_alloc(output), --]
    is(sAlloc) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := msetSlot(cfg.bank_input, alloc = true, row = 1, col = 1)
      io.cmd.bits.slots(1) := msetSlot(cfg.bank_weight, alloc = true, row = 1, col = 1)
      io.cmd.bits.slots(2) := msetSlot(cfg.bank_output, alloc = true, row = 1, col = 4)
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sMvinInput
      }
    }

    // [MVIN_input, MVIN_weight, --, --]
    is(sMvinInput) {
      io.cmd.valid := true.B
      val inputSlot = Mux(
        addrGen.io.isPadding,
        emptySlot(), // Skip MVIN if padding (load zeros implicitly)
        mvinSlot(cfg.bank_input, addrGen.io.inputAddr)
      )
      io.cmd.bits.slots(0) := inputSlot
      io.cmd.bits.slots(1) := mvinSlot(cfg.bank_weight, addrGen.io.weightAddr)
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()
      when(io.cmd.fire) {
        state := sCompute
      }
    }

    // [PRELOAD, COMPUTE, --, --]
    is(sCompute) {
      io.cmd.valid := true.B
      val preSlot = Wire(Valid(new LoopSubCmd(b)))
      preSlot.valid         := true.B
      preSlot.bits          := 0.U.asTypeOf(new LoopSubCmd(b))
      preSlot.bits.cmdType  := LoopSubCmdType.PRELOAD
      preSlot.bits.op1_bank := cfg.bank_weight
      preSlot.bits.wr_bank  := cfg.bank_output
      preSlot.bits.iter     := DIM.U

      val compSlot = Wire(Valid(new LoopSubCmd(b)))
      compSlot.valid             := true.B
      compSlot.bits              := 0.U.asTypeOf(new LoopSubCmd(b))
      compSlot.bits.cmdType      := LoopSubCmdType.COMPUTE
      compSlot.bits.op1_bank     := cfg.bank_weight
      compSlot.bits.op2_bank     := cfg.bank_input
      compSlot.bits.wr_bank      := cfg.bank_output
      compSlot.bits.compute_mode := Mux(isFirstK, 0.U, 1.U) // PRELOADED first, then ACCUMULATED
      compSlot.bits.iter         := DIM.U

      io.cmd.bits.slots(0) := preSlot
      io.cmd.bits.slots(1) := compSlot
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()

      when(io.cmd.fire) {
        when(isLastK) {
          state := sMvout // Need to output results after last k iteration
        }.otherwise {
          advanceIter()
          when(state =/= sFree)(state := sMvinInput)
        }
      }
    }

    // [MVOUT, --, --, --]
    is(sMvout) {
      io.cmd.valid := true.B
      val mvSlot = Wire(Valid(new LoopSubCmd(b)))
      mvSlot.valid          := true.B
      mvSlot.bits           := 0.U.asTypeOf(new LoopSubCmd(b))
      mvSlot.bits.cmdType   := LoopSubCmdType.MVOUT
      mvSlot.bits.bank_id   := cfg.bank_output
      mvSlot.bits.dram_addr := addrGen.io.outputAddr
      mvSlot.bits.iter      := DIM.U

      io.cmd.bits.slots(0) := mvSlot
      io.cmd.bits.slots(1) := emptySlot()
      io.cmd.bits.slots(2) := emptySlot()
      io.cmd.bits.slots(3) := emptySlot()

      when(io.cmd.fire) {
        when(allDone) {
          state := sFree
        }.otherwise {
          advanceIter()
          when(state =/= sFree)(state := sMvinInput)
        }
      }
    }

    // [MSET_free(input), MSET_free(weight), MSET_free(output), --]
    is(sFree) {
      io.cmd.valid         := true.B
      io.cmd.bits.slots(0) := msetSlot(cfg.bank_input, alloc = false, row = 0, col = 0)
      io.cmd.bits.slots(1) := msetSlot(cfg.bank_weight, alloc = false, row = 0, col = 0)
      io.cmd.bits.slots(2) := msetSlot(cfg.bank_output, alloc = false, row = 0, col = 0)
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
