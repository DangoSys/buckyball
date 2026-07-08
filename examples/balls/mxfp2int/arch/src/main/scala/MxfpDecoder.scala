package examples.balls.mxfp2int

import chisel3._
import chisel3.util._

/**
 * MxfpDecoder: pure-combinational decoder from a single MXFP element to INT8.
 *
 * Each implementation defines:
 *   - elemBits: bit-width of one FP element (4 / 6 / 8)
 *   - dequant(raw, scale): convert raw FP element + E8M0 scale to INT8 (signed 8-bit).
 *
 * E8M0 scale (8-bit biased exponent, bias=127):
 *   real_scale = 2^(scale - 127)
 *
 * Output is INT8 with saturation.
 */
trait MxfpDecoder {
  def elemBits: Int

  /**
   * Dequantize one FP element with E8M0 scale.
   * @param raw   FP element bits (elemBits wide)
   * @param scale E8M0 scale byte (8 bits)
   * @return INT8 result (SInt(8.W))
   */
  def dequant(raw: UInt, scale: UInt): SInt
}

object MxfpDecoder {

  /**
   * Shared saturation helper: clamp an SInt to INT8 range.
   */
  def saturateInt8(x: SInt): SInt = {
    val result = Wire(SInt(8.W))
    when(x > 127.S) {
      result := 127.S
    }.elsewhen(x < -128.S) {
      result := -128.S
    }.otherwise {
      result := x(7, 0).asSInt
    }
    result
  }

  /**
   * Shared "signed shift by (scale - 127)" helper.
   * Positive shift = left, negative shift = right (arithmetic).
   */
  def scaleShift(value: SInt, scale: UInt, valueWidth: Int): SInt = {
    val shift    = scale.zext - 127.S(10.W)
    val absShift = Mux(shift >= 0.S, shift, -shift)(4, 0)
    val widened  = value.pad(32)
    Mux(shift >= 0.S, (widened << absShift).asSInt, (widened >> absShift).asSInt)
  }

  /**
   * Pick a decoder implementation by format string.
   */
  def fromFormat(format: String): MxfpDecoder = format match {
    case "MXFP4" => Mxfp4Decoder
    case "MXFP6" => Mxfp6Decoder
    case "MXFP8" => Mxfp8Decoder
    case _       => throw new IllegalArgumentException(s"Unsupported MXFP format: $format")
  }

}
