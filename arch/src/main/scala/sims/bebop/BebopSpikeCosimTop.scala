package sims.bebop

import chisel3._
import chisel3.util.Cat

/** Verilator cosim top: RoCC insn → rd + optional `bankDigestPeek` for future BEMU bank hash compare.
 *
 * `execRet` comes from [[BebopCosimBlocks.execRet]] (per-funct). rd rule matches
 * `bebop/src/emu/iss/iss.rs`: `rd = if v == 0 { funct } else { 0 }`.
 */
class BebopSpikeCosimTop extends RawModule {
  val funct = IO(Input(UInt(7.W)))
  val xs1   = IO(Input(UInt(64.W)))
  val xs2   = IO(Input(UInt(64.W)))

  val result = IO(Output(UInt(64.W)))

  /** 0 until Ball exposes a 64-bit digest; then compare in bebop `vl_worker` when enabled. */
  val bankDigestPeek = IO(Output(UInt(64.W)))

  val execRet = BebopCosimBlocks.execRet(funct, xs1, xs2)
  val known = BebopCosimBlocks.isKnownFunct(funct)
  result := Mux(known && execRet === 0.U, Cat(0.U(57.W), funct), 0.U(64.W))
  bankDigestPeek := BebopCosimBlocks.bankDigestPeek(funct, xs1, xs2)
}
