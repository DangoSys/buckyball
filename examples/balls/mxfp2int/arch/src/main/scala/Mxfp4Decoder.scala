package examples.balls.mxfp2int

import chisel3._
import chisel3.util._

/**
 * Mxfp4Decoder: MXFP4 (4-bit FP element) to INT8 dequantizer.
 *
 * FP4 raw encoding (v1, simplified for testing):
 *   signed_val = raw - 8        (range -8..+7)
 *
 * Result: saturate(signed_val * 2^(scale - 127)) to INT8.
 */
object Mxfp4Decoder extends MxfpDecoder {
  val elemBits = 4

  def dequant(raw: UInt, scale: UInt): SInt = {
    require(raw.getWidth == elemBits, s"Mxfp4Decoder expects $elemBits-bit raw, got ${raw.getWidth}")
    val signedVal = raw.zext - 8.S(6.W) // SInt range -8..+7
    val shifted   = MxfpDecoder.scaleShift(signedVal, scale, 6)
    MxfpDecoder.saturateInt8(shifted)
  }

}
