package examples.balls.mxfp2int

import chisel3._
import chisel3.util._

/**
 * Mxfp6Decoder: MXFP6 (6-bit FP element) to INT8 dequantizer.
 *
 * FP6 raw encoding (v1, simplified for testing):
 *   signed_val = raw - 32       (range -32..+31)
 *
 * Result: saturate(signed_val * 2^(scale - 127)) to INT8.
 */
object Mxfp6Decoder extends MxfpDecoder {
  val elemBits = 6

  def dequant(raw: UInt, scale: UInt): SInt = {
    require(raw.getWidth == elemBits, s"Mxfp6Decoder expects $elemBits-bit raw, got ${raw.getWidth}")
    val signedVal = raw.zext - 32.S(8.W) // SInt range -32..+31
    val shifted   = MxfpDecoder.scaleShift(signedVal, scale, 8)
    MxfpDecoder.saturateInt8(shifted)
  }

}
