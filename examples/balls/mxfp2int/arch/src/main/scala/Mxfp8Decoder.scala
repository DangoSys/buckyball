package examples.balls.mxfp2int

import chisel3._
import chisel3.util._

/**
 * Mxfp8Decoder: MXFP8 (8-bit FP element) to INT8 dequantizer.
 *
 * FP8 raw encoding (v1, simplified for testing):
 *   signed_val = raw - 128      (range -128..+127)
 *
 * Result: saturate(signed_val * 2^(scale - 127)) to INT8.
 */
object Mxfp8Decoder extends MxfpDecoder {
  val elemBits = 8

  def dequant(raw: UInt, scale: UInt): SInt = {
    require(raw.getWidth == elemBits, s"Mxfp8Decoder expects $elemBits-bit raw, got ${raw.getWidth}")
    val signedVal = raw.zext - 128.S(10.W) // SInt range -128..+127
    val shifted   = MxfpDecoder.scaleShift(signedVal, scale, 10)
    MxfpDecoder.saturateInt8(shifted)
  }

}
