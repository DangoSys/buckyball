package sims.bebop

import chisel3._
import chisel3.util._

/** Per-`funct` hooks for Spike↔Verilator cosim. Keep literals in sync with `bebop/src/emu/inst/decode.rs`. */
object BebopCosimBlocks {

  // FUNCT_* (7-bit RoCC custom field)
  val F_MSET:  UInt = 32.U(7.W)
  val F_MVIN:  UInt = 33.U(7.W)
  val F_MVOUT: UInt = 16.U(7.W)
  val F_FENCE: UInt = 0.U(7.W)

  /** Mirrors `decode::execute_known` inner `u64` return (`ret` before iss maps rd). */
  def execRet(funct: UInt, xs1: UInt, xs2: UInt): UInt = {
    val _ = (xs1, xs2)
    MuxLookup(
      funct,
      0.U(64.W),
    )(
      Seq(
        F_FENCE  -> 0.U(64.W),
        F_MSET   -> 0.U(64.W),
        F_MVIN   -> 0.U(64.W),
        F_MVOUT  -> 0.U(64.W),
        48.U(7.W) -> 0.U(64.W), // FUNCT_IM2COL
        49.U(7.W) -> 0.U(64.W), // FUNCT_TRANSPOSE
        64.U(7.W) -> 0.U(64.W), // FUNCT_MUL_WARP16
      ),
    )
  }

  /** Optional observable for bank difftest; 0 = not implemented. Wire Ball SRAM hash later. */
  def bankDigestPeek(funct: UInt, xs1: UInt, xs2: UInt): UInt = {
    val _ = (funct, xs1, xs2)
    0.U(64.W)
  }
}
