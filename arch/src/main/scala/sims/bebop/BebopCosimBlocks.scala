package sims.bebop

import chisel3._
import chisel3.util._

/** Per-`funct` hooks for Spike↔Verilator cosim. Keep literals in sync with `bebop/src/emu/inst/decode.rs`. */
object BebopCosimBlocks {

  // FUNCT_* (7-bit RoCC custom field)
  val F_FENCE:                             UInt = 0.U(7.W)
  val F_BARRIER:                           UInt = 1.U(7.W)
  val F_GEMMINI_CONFIG:                    UInt = 2.U(7.W)
  val F_GEMMINI_FLUSH:                     UInt = 3.U(7.W)
  val F_BDB_COUNTER:                       UInt = 4.U(7.W)
  val F_MVOUT:                             UInt = 16.U(7.W)
  val F_MSET:                              UInt = 32.U(7.W)
  val F_MVIN:                              UInt = 33.U(7.W)
  val F_IM2COL:                            UInt = 48.U(7.W)
  val F_TRANSPOSE:                         UInt = 49.U(7.W)
  val F_RELU:                              UInt = 50.U(7.W)
  val F_QUANT:                             UInt = 51.U(7.W)
  val F_DEQUANT:                           UInt = 52.U(7.W)
  val F_GEMMINI_PRELOAD:                   UInt = 53.U(7.W)
  val F_BDB_BACKDOOR:                      UInt = 54.U(7.W)
  val F_MUL_WARP16:                        UInt = 64.U(7.W)
  val F_BFP:                               UInt = 65.U(7.W)
  val F_GEMMINI_COMPUTE_PRELOADED:         UInt = 66.U(7.W)
  val F_GEMMINI_COMPUTE_ACCUMULATED:       UInt = 67.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_BOUNDS:     UInt = 80.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_ADDR_A:     UInt = 81.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_ADDR_B:     UInt = 82.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_ADDR_D:     UInt = 83.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_ADDR_C:     UInt = 84.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_STRIDES_AB: UInt = 85.U(7.W)
  val F_GEMMINI_LOOP_WS_CONFIG_STRIDES_DC: UInt = 86.U(7.W)
  val F_GEMMINI_LOOP_WS:                   UInt = 87.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_1:     UInt = 96.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_2:     UInt = 97.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_3:     UInt = 98.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_4:     UInt = 99.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_5:     UInt = 100.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_6:     UInt = 101.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_7:     UInt = 102.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_8:     UInt = 103.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS_CONFIG_9:     UInt = 104.U(7.W)
  val F_GEMMINI_LOOP_CONV_WS:              UInt = 105.U(7.W)

  val knownFuncts: Seq[UInt] = Seq(
    F_FENCE,
    F_BARRIER,
    F_GEMMINI_CONFIG,
    F_GEMMINI_FLUSH,
    F_BDB_COUNTER,
    F_MVOUT,
    F_MSET,
    F_MVIN,
    F_IM2COL,
    F_TRANSPOSE,
    F_RELU,
    F_QUANT,
    F_DEQUANT,
    F_GEMMINI_PRELOAD,
    F_BDB_BACKDOOR,
    F_MUL_WARP16,
    F_BFP,
    F_GEMMINI_COMPUTE_PRELOADED,
    F_GEMMINI_COMPUTE_ACCUMULATED,
    F_GEMMINI_LOOP_WS_CONFIG_BOUNDS,
    F_GEMMINI_LOOP_WS_CONFIG_ADDR_A,
    F_GEMMINI_LOOP_WS_CONFIG_ADDR_B,
    F_GEMMINI_LOOP_WS_CONFIG_ADDR_D,
    F_GEMMINI_LOOP_WS_CONFIG_ADDR_C,
    F_GEMMINI_LOOP_WS_CONFIG_STRIDES_AB,
    F_GEMMINI_LOOP_WS_CONFIG_STRIDES_DC,
    F_GEMMINI_LOOP_WS,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_1,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_2,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_3,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_4,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_5,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_6,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_7,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_8,
    F_GEMMINI_LOOP_CONV_WS_CONFIG_9,
    F_GEMMINI_LOOP_CONV_WS
  )

  def isKnownFunct(funct: UInt): Bool = knownFuncts.map(_ === funct).reduce(_ || _)

  /** Mirrors `decode::execute_known` inner `u64` return (`ret` before iss maps rd). */
  def execRet(funct: UInt, xs1: UInt, xs2: UInt): UInt = {
    val _ = (xs1, xs2)
    MuxLookup(
      funct,
      0.U(64.W)
    )(
      Seq(
        F_FENCE                             -> 0.U(64.W),
        F_BARRIER                           -> 0.U(64.W),
        F_GEMMINI_CONFIG                    -> 0.U(64.W),
        F_GEMMINI_FLUSH                     -> 0.U(64.W),
        F_BDB_COUNTER                       -> 0.U(64.W),
        F_MVOUT                             -> 0.U(64.W),
        F_MSET                              -> 0.U(64.W),
        F_MVIN                              -> 0.U(64.W),
        F_IM2COL                            -> 0.U(64.W),
        F_TRANSPOSE                         -> 0.U(64.W),
        F_RELU                              -> 0.U(64.W),
        F_QUANT                             -> 0.U(64.W),
        F_DEQUANT                           -> 0.U(64.W),
        F_GEMMINI_PRELOAD                   -> 0.U(64.W),
        F_BDB_BACKDOOR                      -> 0.U(64.W),
        F_MUL_WARP16                        -> 0.U(64.W),
        F_BFP                               -> 0.U(64.W),
        F_GEMMINI_COMPUTE_PRELOADED         -> 0.U(64.W),
        F_GEMMINI_COMPUTE_ACCUMULATED       -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_BOUNDS     -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_ADDR_A     -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_ADDR_B     -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_ADDR_D     -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_ADDR_C     -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_STRIDES_AB -> 0.U(64.W),
        F_GEMMINI_LOOP_WS_CONFIG_STRIDES_DC -> 0.U(64.W),
        F_GEMMINI_LOOP_WS                   -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_1     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_2     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_3     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_4     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_5     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_6     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_7     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_8     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS_CONFIG_9     -> 0.U(64.W),
        F_GEMMINI_LOOP_CONV_WS              -> 0.U(64.W)
      )
    )
  }

  /** Optional observable for bank difftest; 0 = not implemented. Wire Ball SRAM hash later. */
  def bankDigestPeek(funct: UInt, xs1: UInt, xs2: UInt): UInt = {
    val _ = (funct, xs1, xs2)
    0.U(64.W)
  }

}
